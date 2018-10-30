rng(0);
scales = [8, 16, 32, 64];
normH = 16;
normW = 16;
%bowCs = HW5_BoW.learnDictionary(scales, normH, normW);

%[trIds, trLbs] = ml_load('../bigbangtheory_v2/train.mat',  'imIds', 'lbs');             
%tstIds = ml_load('../bigbangtheory_v2/test.mat', 'imIds'); 
tstLbs = [];
for i = 1:length(tstIds)
    tstLbs = [tstLbs; double(rand())];
end

%trD  = HW5_BoW.cmpFeatVecs(trIds, scales, normH, normW, bowCs);
%trD = trD';
%tstD = HW5_BoW.cmpFeatVecs(tstIds, scales, normH, normW, bowCs);
%tstD = tstD';

gamma = 0.4780;
%[trainK, testK] = cmpExpX2Kernel2(trD, tstD, gamma);
options = sprintf('-c 10 -g %d -t 4', gamma);
%model = svmtrain(trLbs, trainK, options);
[predict_label] = svmpredict(tstLbs, testK, model);
csvwrite('predTestLabels.csv', predict_label);

