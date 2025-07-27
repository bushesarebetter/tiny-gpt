import torch

from tinygpt import GPT


def small(**kw):
    torch.manual_seed(0)
    return GPT(vocab_size=20, block_size=16, n_layer=2, n_head=2, d=32, dropout=0.0, **kw)


def test_causal_mask():
    m = small().eval()
    x = torch.randint(20, (2, 16))
    y = x.clone()
    y[:, 8:] = (y[:, 8:] + 1) % 20  # change only the future
    lx, _ = m(x)
    ly, _ = m(y)
    assert torch.allclose(lx[:, :8], ly[:, :8], atol=1e-6)
    assert not torch.allclose(lx[:, 8:], ly[:, 8:])


def test_loss_decreases():
    for tie in (True, False):
        m = small(tie=tie)
        x = torch.randint(20, (4, 16))
        y = torch.roll(x, -1, dims=1)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        losses = []
        for _ in range(30):
            _, loss = m(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < 0.7 * losses[0]


def test_sample_length():
    m = small().eval()
    prompt = torch.tensor([[1, 2, 3]])
    out = m.generate(prompt, 40, temperature=0.8, top_k=5)  # longer than block_size, exercises cropping
    assert out.shape == (1, 43)
    assert torch.equal(out[:, :3], prompt)
    assert out.max() < 20
