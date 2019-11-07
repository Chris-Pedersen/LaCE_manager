import p1d_arxiv

rescalings=True

archive=p1d_arxiv.ArxivP1D()

archive.plot_3D_samples("T0","gamma","mF",tau_scalings=False,temp_scalings=False)
archive.plot_3D_samples("kF_Mpc","mF","Delta2_p",tau_scalings=False,temp_scalings=False)