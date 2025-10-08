# Cosmology-project
These are the codes used to create the figures for my first cosmology project. It solves the Boltzmann hierarchy equations using CAMB for a hybrid dark sector model explores, mostly, the background evolution of various components and the dark energy equation of state. It also provides tools for constraining the model's parameters using MCMC.

The modified CAMB version I am using can be found in https://github.com/joaoreboucas1/CAMB-HDS.git and it is based on the model presented in https://arxiv.org/abs/2211.13653. To run the files it is necessary to clone the repository and compile both the fortran file and the camb file. For windows users I suggest employing a subsystem for Linux (like WSL).

The MCMC analysis was carried out with Cobaya, with likelihoods including Planck 2018, DESI BAO, and PantheonPlus Supernova data. The chains were analyzed using the GetDist library.

There can be found here many other files exploring the numerical solutions of the same system using an ODEINT method, which is much faster and more useful to explore the parameter space of the model without the need of modyfing CAMB's source code all the time. I am currently implementing different potentials into the Klein-Gordon equations for the DE field. 
