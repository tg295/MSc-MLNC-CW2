%% %% %% Q1 MULTI-LAYER PERCEPTRON %%%%%%%%%%%
clear all
close all
%% Load Data
load Activities.mat
%% 3D scatter plot 
train_comb=[train_data(:,1) train_data(:,2) train_data(:,3) train_labels(:,1)];
%splitting data into classes for visualisation
train_class1=train_comb((train_comb(:,4)==1),:);
train_class2=train_comb((train_comb(:,4)==2),:);
train_class3=train_comb((train_comb(:,4)==3),:);
train_class4=train_comb((train_comb(:,4)==4),:);

figure
scatter3(train_class1(:,1),train_class1(:,2),train_class1(:,3),9,'r','.');
hold on 
scatter3(train_class2(:,1),train_class2(:,2),train_class2(:,3),9,'b','.');
hold on 
scatter3(train_class3(:,1),train_class3(:,2),train_class3(:,3),9,'g','.');
hold on 
scatter3(train_class4(:,1),train_class4(:,2),train_class4(:,3),9,'m','.');
xlabel('x [unit-distance/s^2]')
ylabel('y [unit-distance/s^2]')
zlabel('z [unit-distance/s^2]')
legend('walking','jogging','walking upstairs','walking downstairs')
%% %% Q1(b) Investigating network geometry 
%This section shows the code used to investigate different network
%geometries; number of layers/number of neurons in each layer. The
%algorithm is set up using a for loop so that multiple configurations can
%be investigated at once and the resulting accuracies plotted on a single
%graph. Here the code is set up for the final chosen configuration: a single layer of 20 neurons. 
nbrOfEpochs_max = 500;
nbrOfEpochs=[(1:nbrOfEpochs_max)'];
% initialise variables 
% "multi" variables are to allow data to be stored from multiple configurations 
nbrNeuronsMulti=[];
accuracyMulti=[];
best_prediction_multi=[];
accuracy_max=[];
j={[20]};%j can be set as multiple configurations e.g.{[2],[5],[10,10]}
for i=1:size(j,2);
    tic;%recording time taken for each configuration to run
nbrOfNeuronsInEachHiddenLayer = j{1,i};

[accuracy, best_prediction] = MLP_REST(train_data, train_labels, test_data, test_labels, nbrOfNeuronsInEachHiddenLayer, nbrOfEpochs_max);
 
nbrNeuronsMulti{1,i}= nbrOfNeuronsInEachHiddenLayer; 
accuracyMulti(nbrOfEpochs,i)=accuracy;
best_prediction_multi((1:length(best_prediction)),i)=best_prediction;
accuracy_max(1,i)=sum(accuracyMulti((476:500),i))/length(accuracyMulti(476:500));%maximum accuracy is taken as the average over the last 25 epochs
toc; 
time(1,i)=toc;
end 

% Plot of accuracy over number of epochs for multiple configurations
figure
plot(nbrOfEpochs,accuracyMulti);
xlabel('number of epochs')
ylabel('classification accuracy [%]')
title('Learning curve for 20 neurons single-layer')
legend(strcat('no. of Neurons =','[20]'),'Location','southeast')

%% %% Q1(c)Final configuration & confusion matrix 
% Best MLP config: 1 layer, 20 neurons
% -> accuracy = 0.627, comp time = 175 seconds
%%% Confusion matrix %%%
%number of weights
nbrOfweights=((3+1)*nbrOfNeuronsInEachHiddenLayer)+((1+nbrOfNeuronsInEachHiddenLayer)*4);
%confusion matrix
label_compare=[test_labels(:,1),best_prediction(:,1)];
mat=zeros(4);

for i=1:4
    for j=1:4
    mat(i,j)=sum(label_compare(:,1)==i & label_compare(:,2)==j)
    end 
end

%Precision for each class - probabilities for each class being predicted correctly  
prob=(diag(mat)'./sum(mat'))*100;
%%%%%%%%%%%%% Class 1 and 2 are easiest to distinguish %%%%%%%%%%%%%

%% %% %% Q2 BINARY CLASSIFICATION %%%%%%%%%%%%
clear all
close all
%% Load Data
load Activities.mat
%% Split into two classes (1 & 2)
train_data_comb=[train_data,train_labels];
train_data_binary=train_data_comb((train_data_comb(:,4)==1 | train_data_comb(:,4)==2),1:3);%binary training data accelerations 
train_labels_binary=train_data_comb((train_data_comb(:,4)==1 | train_data_comb(:,4)==2),4);%binary training data classes 

test_data_comb=[test_data,test_labels];
test_data_binary=test_data_comb((test_data_comb(:,4)==1 | test_data_comb(:,4)==2),1:3);%binary test data accelerations 
test_labels_binary=test_data_comb((test_data_comb(:,4)==1 | test_data_comb(:,4)==2),4);%binary test data classes 
%% MLP for binary classification
%Here the code is the same as Q1(b) and with the same configuration. Inputs
%are just changed to the binary data variables.
nbrOfEpochs_max = 500;
nbrOfEpochs=[(1:nbrOfEpochs_max)'];

nbrNeuronsMulti=[];
accuracyMulti=[];
best_prediction_multi=[];
accuracy_max=[];
j={[20]};
for i=1:size(j,2);
    tic;
nbrOfNeuronsInEachHiddenLayer = j{1,i};

[accuracy, best_prediction] = MLP_REST(train_data_binary, train_labels_binary, test_data_binary, test_labels_binary, nbrOfNeuronsInEachHiddenLayer, nbrOfEpochs_max);
 
nbrNeuronsMulti{1,i}= nbrOfNeuronsInEachHiddenLayer; 
accuracyMulti(nbrOfEpochs,i)=accuracy;
best_prediction_multi((1:length(best_prediction)),i)=best_prediction;
accuracy_max(1,i)=sum(accuracyMulti((476:500),i))/length(accuracyMulti(476:500));
toc; 
time(1,i)=toc;
end 

figure
plot(nbrOfEpochs,accuracyMulti);
xlabel('number of epochs')
ylabel('classification accuracy [%]')
title('Learning curve for 20 neurons single-layer - Binary classification')
legend(strcat('no. of Neurons =','[20]'),'Location','southeast')

%% Confusion matrix - binary classification 
label_compare_binary=[test_labels_binary(:,1),best_prediction(:,1)];
mat_binary=zeros(2);

for i=1:2
    for j=1:2
    mat_binary(i,j)=sum(label_compare_binary(:,1)==i & label_compare_binary(:,2)==j)
    end 
end

%probabilities for each class being predicted correctly 
prob_binary=(diag(mat_binary)'./sum(mat_binary'))*100;

%% %% %% Q3 & 4 MY CLASSIFIER %%%%%%%%%%%%%%

%% Train my classifier 
% *Training does not actually occur here as using k-NN, therefore training
% occurs within ClassifyX. Here the function is just used to set parameters as the training
% datapoints and their respective classes.
parameters = TrainClassifierX(train_data_binary,train_labels_binary);

%% Classification using k-NN
tic;
class = ClassifyX(test_data_binary, parameters)
accuracy_kNN = sum(class(:,1)==test_labels_binary)/length(class);%Accuracy of my classifier
toc;
time_kNN=toc;
%% Confusion matrix - k-NN 
label_compare_binary_kNN=[test_labels_binary(:,1),class(:,1)];
mat_binary_kNN=zeros(2);

for i=1:2
    for j=1:2
    mat_binary_kNN(i,j)=sum(label_compare_binary_kNN(:,1)==i & label_compare_binary_kNN(:,2)==j)
    end 
end

%probabilities for each class being right 
prob_binary_kNN=(diag(mat_binary_kNN)'./sum(mat_binary_kNN'))*100;

%% Sanity check
SanityCheck()
%% 




