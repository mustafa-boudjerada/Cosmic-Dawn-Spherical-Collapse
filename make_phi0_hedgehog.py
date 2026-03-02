import argparse, os
import torch

def main():
    ap = argparse.ArgumentParser(description="Generate hedgehog phi0_N*.pt on S^3 (Phi·Phi=1).")
    ap.add_argument("--N", type=int, required=True, help="grid size (e.g. 300)")
    ap.add_argument("--L", type=float, required=True, help="half box length, coordinates in [-L,L]")
    ap.add_argument("--R0", type=float, default=1.0, help="core radius parameter in profile f(r)")
    ap.add_argument("--p", type=float, default=2.0, help="profile exponent in f(r)=2 arctan((R0/r)^p)")
    ap.add_argument("--dtype", type=str, default="float64", choices=["float32","float64"])
    ap.add_argument("--out", type=str, default=None, help="output path (default outputs/phi0_cache/phi0_N{N}.pt)")
    args = ap.parse_args()

    dt = torch.float64 if args.dtype=="float64" else torch.float32
    N=args.N
    xs = torch.linspace(-args.L, args.L, N, dtype=dt)
    X,Y,Z = torch.meshgrid(xs,xs,xs, indexing="ij")
    r = torch.sqrt(X*X+Y*Y+Z*Z) + 1e-12
    f = 2.0 * torch.atan((args.R0 / r)**args.p)
    s = torch.sin(f)
    c = torch.cos(f)
    nx,ny,nz = X/r, Y/r, Z/r

    phi = torch.empty((N,N,N,4), dtype=dt)
    phi[...,0]=c
    phi[...,1]=s*nx
    phi[...,2]=s*ny
    phi[...,3]=s*nz

    phi = phi / torch.linalg.norm(phi, dim=-1, keepdim=True)

    out = args.out
    if out is None:
        out = os.path.join("outputs","phi0_cache",f"phi0_N{N}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(phi, out)

    n = torch.linalg.norm(phi, dim=-1)
    print("saved", out, phi.shape, "min|phi|", float(n.min()), "max|phi|", float(n.max()))

if __name__ == "__main__":
    main()
