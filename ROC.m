%M = dlmread('E:\Benutzer\Daniel\Desktop\test.csv', '\t')
M = dlmread('E:\Benutzer\Daniel\Desktop\UbuntuShare\_result1.csv', '\t')
M2 = dlmread('E:\Benutzer\Daniel\Desktop\UbuntuShare\_result2.csv', '\t')

M = M.'
targets = M(1,:)
outputs = M(2,:)

M2 = M2.'
targets2 = M2(1,:)
outputs2 = M2(2,:)

plotroc(targets, outputs, 'Sliding Window', targets2, outputs2, 'Merged Contour')



%run('VLFEATROOT/toolbox/vl_setup')
%[recall, precision] = VL_PR(targets, outputs)