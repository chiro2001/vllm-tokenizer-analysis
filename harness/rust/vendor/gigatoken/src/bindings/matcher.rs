//! A prebuilt multi-substring matcher exposed to Python, used by the
//! tiktoken compat layer to scan documents for special tokens. The scan runs fused into
//! the encode_batch_list call (`scan_docs`), in parallel over documents
//! with the GIL released, instead of one Python-level substring search per
//! token per document.

use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;

pyo3::create_exception!(
    gigatoken_rs,
    SpecialTokenFound,
    pyo3::exceptions::PyException,
    "A forbidden pattern was found while encoding; args[0] holds the sorted \
     indices of every matched pattern, for the caller to raise its own error."
);

/// Compiled multi-pattern substring matcher. Build once from a list of
/// patterns, then query which patterns occur in a text — plain substring
/// containment, exactly like `pattern in text` for each pattern.
/// Underscore-named on the Python side: an implementation detail of the
/// compat layer, not part of the public API.
#[pyclass(frozen, name = "_SubstringMatcher")]
pub(crate) struct SubstringMatcher {
    automaton: aho_corasick::AhoCorasick,
    n_patterns: usize,
}

impl SubstringMatcher {
    /// Which patterns occur in `haystack`, as a bitvec over pattern indices.
    /// Overlapping search keeps containment semantics exact when one
    /// pattern is a substring of another (which the default Standard match
    /// kind supports).
    fn scan(&self, haystack: &[u8]) -> Vec<bool> {
        let mut seen = vec![false; self.n_patterns];
        let mut found = 0;
        for m in self.automaton.find_overlapping_iter(haystack) {
            let idx = m.pattern().as_usize();
            if !seen[idx] {
                seen[idx] = true;
                found += 1;
                if found == self.n_patterns {
                    break;
                }
            }
        }
        seen
    }

    /// Scan every document — in parallel with rayon when `parallel` is set;
    /// serially otherwise (forked workers must never touch the global pool).
    /// Ok(()) when no pattern occurs anywhere; SpecialTokenFound carrying
    /// the sorted indices of every matched pattern otherwise. Call with the
    /// GIL released.
    pub(crate) fn scan_docs(&self, docs: &[&[u8]], parallel: bool) -> PyResult<()> {
        use rayon::prelude::*;
        let merge = |mut a: Vec<bool>, b: Vec<bool>| {
            a.iter_mut().zip(&b).for_each(|(x, y)| *x |= y);
            a
        };
        let seen = if parallel {
            docs.par_iter()
                .map(|d| self.scan(d))
                .reduce(|| vec![false; self.n_patterns], merge)
        } else {
            docs.iter()
                .map(|d| self.scan(d))
                .fold(vec![false; self.n_patterns], merge)
        };
        let hits: Vec<usize> = seen
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| s.then_some(i))
            .collect();
        if hits.is_empty() {
            Ok(())
        } else {
            Err(SpecialTokenFound::new_err(hits))
        }
    }
}

#[pymethods]
impl SubstringMatcher {
    #[new]
    fn new(patterns: Vec<String>) -> PyResult<Self> {
        let n_patterns = patterns.len();
        let automaton = aho_corasick::AhoCorasick::new(&patterns).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "failed to build matcher: {e}"
            ))
        })?;
        Ok(Self {
            automaton,
            n_patterns,
        })
    }

    /// Sorted indices of the patterns that occur in `text`. Runs with the
    /// GIL released.
    fn present(&self, py: Python<'_>, text: PyBackedStr) -> Vec<usize> {
        py.detach(|| {
            self.scan(text.as_bytes())
                .iter()
                .enumerate()
                .filter_map(|(i, &s)| s.then_some(i))
                .collect()
        })
    }

    fn __len__(&self) -> usize {
        self.n_patterns
    }
}
