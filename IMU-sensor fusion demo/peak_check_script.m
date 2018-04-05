load('az_gx.mat')
figure; plot(ax);
y=ax(100:400,:);
plot(y);
[maxtab, mintab] = peakdet(y, 0.5);
hold on; plot(mintab(:,1), mintab(:,2), 'g*');
plot(maxtab(:,1), maxtab(:,2), 'r*')

[pksY,plocY] = findpeaks(y)
plot(plocY,pksY, 'b+')