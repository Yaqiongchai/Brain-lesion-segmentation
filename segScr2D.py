import segF2D as s
namelist = ['RIS', 'TOY', 'DEB', 'AAT']

for ni in namelist:
        s.seg8('prediction/'+ni+'histPadAXI.nii.gz',8)
        s.seg16('prediction/'+ni+'histPadAXI.nii.gz',16)
        s.seg32('prediction/'+ni+'histPadAXI.nii.gz',32)
        s.seg64('prediction/'+ni+'histPadAXI.nii.gz',64)
        s.seg128('prediction/'+ni+'histPadAXI.nii.gz',128)
