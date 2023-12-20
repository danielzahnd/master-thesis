# nf-milne-eddington-example-6
This code experiments with normalizing flows used to invert Milne-Eddington atmospheres. In this case, a single map of penumbra formation maps (Stokes parameters at different wavelengths for each pixel) is considered for training. This map is inverted using a Milne-Eddington algorithm. A normalizing flow of the affine coupling layer type is trained on this data (observations and corresponding inverted data). On samples of a prior unseen map of data of the same penumbra formation dataset, the normalizing flow is then tested and compared to the corresponding result of an MCMC inversion using the Milne-Eddington model.

Used data for this experiment can be found at [penumbra-formation-maps](https://drive.google.com/drive/folders/1-W3vCJC4gEsQWW0pzwF8PbQ3erE0eGPI?usp=drive_link/) and [sunspot-map](https://drive.google.com/drive/folders/1AM6oA1mLYQ_DtIlSv52aYXDNDTygRQyq?usp=drive_link).