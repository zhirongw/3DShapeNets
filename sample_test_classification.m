function [samples] = sample_test_classification(model,class)
% multiple category gibbs sampling from the model with a class specified.
% Do gibbs sampling on the top associative memeory, and propagate it down.

fprintf('sampling %s from top RBM\n', model.classnames{class});
global kConv_backward kConv_backward_c;

if isfield(model.layers{2}, 'w')
    model = merge_model(model);
end
n = 32;

num_layer = length(model.layers);
hn = 0.5 * rand([n, model.layers{num_layer}.layerSize],'single');
hn_1 = 0.5 * rand([n, prod(model.layers{num_layer-1}.layerSize)],'single');

label = 0.5 * ones(n, model.classes);
if exist('class', 'var')
    label = zeros(n, model.classes);
    label(:, class) = 1;
end

if ~isfield(model ,'duplicate')
    model.duplicate = 1;
end
temp_w = model.layers{num_layer}.w;
temp_w(1:model.classes,:) = temp_w(1:model.classes,:) * model.duplicate;
for i = 1 : 2000
	% alternating gibbs
    % prop up
    hn = bsxfun(@plus, [label, hn_1] * temp_w, model.layers{num_layer}.c);
    hn = 1 ./ (1 + exp(-hn));
    hn = single(hn > rand(size(hn)));

    % prop down
    hn_1 = bsxfun(@plus, hn * model.layers{num_layer}.w' , model.layers{num_layer}.b);
    if ~exist('class','var')
        temp_exponential = exp(bsxfun(@minus,hn_1(:, 1:model.classes),max(hn_1(:, 1:model.classes),[],2)));
        label = bsxfun(@rdivide, temp_exponential, sum(temp_exponential,2)); 
        label = mnrnd(1,label);
    end
    hn_1 = 1 ./ ( 1 + exp(-hn_1));
    hn_1 = single(hn_1 > rand(size(hn_1)));
    hn_1 = hn_1(:,model.classes+1:end);
end

samples = hn_1;

for l = num_layer - 1 : -1 : 2
    if l == 2
        samples = reshape(samples, [n, model.layers{l}.layerSize]);
        presigmoid = myConvolve(kConv_backward, samples, model.layers{l}.dw, model.layers{l}.stride, 'backward');
        presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
        samples = 1 ./ ( 1 + exp(-presigmoid));
    elseif l > 2 && strcmp(model.layers{l}.type, 'convolution')
        samples = reshape(samples, [n, model.layers{l}.layerSize]);
        presigmoid = myConvolve(kConv_backward_c, samples, model.layers{l}.dw, model.layers{l}.stride, 'backward');
        presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
        samples = 1 ./ ( 1 + exp(-presigmoid));
    else
        samples = reshape(samples, [n, model.layers{l}.layerSize]);
        samples = bsxfun(@plus, samples * model.layers{l}.uw' , model.layers{l}.b);
        samples = 1 ./ ( 1 + exp(-samples));
    end
    
end
