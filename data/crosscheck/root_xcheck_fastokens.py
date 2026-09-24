import sys; sys.path.insert(0, "/workspace/fastokens_pkg")
import time, json
from tokenizers import Tokenizer
path = "/models/Qwen3-0.6B/tokenizer.json"
text_s = "The quick brown fox jumps over the lazy dog. 这是一段用于测试的中文文本。" * 8
text_l = text_s * 8
tok = Tokenizer.from_file(path)
ids = tok.encode(text_s).ids

def bench(fn, n=200, warm=20):
    for _ in range(warm): fn()
    s = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - s) / n * 1e6

out = {"n_tokens": len(ids), "n_chars": len(text_s)}
out["hf_encode_s"] = bench(lambda: tok.encode(text_s))
out["hf_encode_l"] = bench(lambda: tok.encode(text_l), n=50)

import fastokens
from fastokens._compat import _TokenizerShim
shim = _TokenizerShim(tok)
ids_f = shim.encode(text_s, add_special_tokens=False).ids
out["ids_identical"] = (ids_f == ids)
out["fastokens_encode_s"] = bench(lambda: shim.encode(text_s, add_special_tokens=False))
out["fastokens_encode_l"] = bench(lambda: shim.encode(text_l, add_special_tokens=False), n=50)
out["hf_decode_s"] = bench(lambda: tok.decode(ids))
out["fastokens_decode_s"] = bench(lambda: shim.decode(ids))
print("RESULT " + json.dumps({k: (round(v,2) if isinstance(v,float) else v) for k,v in out.items()}))
