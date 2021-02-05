# CNN_kernel_conversion for CT images
CNN caffe pre-trained network for converting between B30 (1S1) and B70 (1L1) and vice-versa

The credit for the CNN model goes to Choe et al (2019).
The original github post with the source code and instructions can be found at https://github.com/leegaeun/CT_Kernel_Conversion [1]

The pretrained network was trained using repeat CT Lung images (kernel B70) that are publicly available on the TCIA at https://wiki.cancerimagingarchive.net/display/Public/RIDER+Lung+CT#2251273279368e51be0f4512a5934daff0cfe302 [2]
and their counterpart images (kernel B30), which are being prepared for upload to the TCIA.


References
1. Choe, J., et al. (2019). "Deep Learning–based Image Conversion of CT Reconstruction Kernels Improves Radiomics Reproducibility for Pulmonary Nodules or Masses." Radiology 292(2): 365-373.
2. Zhao, Binsheng, Schwartz, Lawrence H, & Kris, Mark G. (2015). Data From RIDER_Lung CT. The Cancer Imaging Archive. DOI: 10.7937/K9/TCIA.2015.U1X8A5NR
3. Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057.  DOI: 10.1007/s10278-013-9622-7

