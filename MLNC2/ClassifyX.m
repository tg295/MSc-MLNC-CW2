%% Theo Bacon Gardner, CID: 1439118
function class = ClassifyX(input, parameters)
%All implementaion should be inside the function
%parameters{1,1}=training data 
%parameters{1,2}=training labels
k=5;%% number of neighbours taken into consideration 
% initialise variables
distance=zeros(size(parameters{1,2},1), size(input,1));%% matrix to store distance values
class=zeros(length(k), 1);%% matrix to store predicted class values
for i=1:size(input,1)%% run over all datapoints in the test data 
    for j=1:size(parameters{1,1},1)%% for every test datapoint run over all training datapoints
        distance(j,i)=norm(parameters{1,1}(j,:)-input(i,:));%% calculate distance of every training datapoint from each testdatapoint
    end
    [~,idx]=sort(distance, 'ascend');%% sort the distances of every training data point from each test datapoint into ascending order
    closest_train_labels=parameters{1,2}(idx(1:k,i));%%select the k closest training datapoints from each test datapoint
    class(i,:)=mode(closest_train_labels);%% assign the predicted class for each test datapoint as the most commonly occuring class of the k nearest training datapoints
end
end