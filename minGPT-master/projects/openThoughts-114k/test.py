import torch
from mingpt.model import GPT
from mingpt.utils import CfgNode as CN
from mingpt.bpe import BPETokenizer

CKPT = "./out/gpt2_medium_reasoning/model_final.pt"

def get_config():
    C = CN()
    C.model = GPT.get_default_config()
    C.model.model_type = "gpt2-medium"
    C.model.vocab_size = 50257
    C.model.block_size = 255  # wie beim Training
    return C

@torch.no_grad()
@torch.no_grad()
def main():
    tok = BPETokenizer()
    config = get_config()
    model = GPT(config.model)

    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = "User: Hello.\n\nAssistant:"
    ids = tok(prompt)
    if isinstance(ids, torch.Tensor):
        ids = ids.view(-1).tolist()
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    out = model.generate(idx, max_new_tokens=120, do_sample=True, temperature=0.9, top_k=50)

    # decode expects 1D tensor
    print(tok.decode(out[0].detach().cpu()))


if __name__ == "__main__":
    main()
