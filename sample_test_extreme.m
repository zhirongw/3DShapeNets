function [batch_data, batch_label] = sample_test_extreme(model, class)
% Gibbs sampling for multi-class models. Somewhat like
% sample_test_classification, but sampling process involves all layers
% in way that mimics the completion process(up down up down). If this
% sampling can give good results, completion performance is more likely to
% be better.

addpath 3D;
global kConv_forward2 kConv_forward_c kConv_backward kConv_backward_c;
fprintf('sampling %s from top RBM\n', model.classnames{class});

if ~isfield(model.layers{2},'uw')
    model  = merge_model(model);
end

param = [];
param.epochs = 100;
param.gibbs_iter = 1;
num_layer = length(model.layers);
batch_size = 32;
batch_data = zeros([batch_size, model.layers{1}.layerSize]);
batch_data = single(rand(size(batch_data)) > 0.5);
batch_label = zeros(batch_size, model.classes);
if exist('class','var')
    batch_label(:,class) = 1;
else
    batch_label = 0.5 * ones(batch_size, model.classes);
end

for epoch = 1 : param.epochs
    % propagate/inference bottum up using recognition weight. 
    for l = 2 : num_layer - 1
        if l == 2
            hidden_presigmoid = myConvolve2(kConv_forward2, batch_data, model.layers{l}.uw, model.layers{l}.stride, 'forward');
            hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
            batch_hidden_prob = sigmoid(hidden_presigmoid);
        elseif strcmp(model.layers{l}.type, 'convolution')
            hidden_presigmoid = myConvolve(kConv_forward_c, batch_data, model.layers{l}.uw, model.layers{l}.stride, 'forward');
            hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
            batch_hidden_prob = sigmoid(hidden_presigmoid);
        else
            batch_data = reshape(batch_data, batch_size, []);
            hidden_presigmoid = bsxfun(@plus, ...
                batch_data * model.layers{l}.uw, model.layers{l}.c);
            batch_hidden_prob = 1 ./ ( 1 + exp(- hidden_presigmoid) );
        end
        batch_data = batch_hidden_prob;
    end

    batch_data = reshape(batch_data, batch_size, []);

    hn_1 = batch_data;

    temp_w = model.layers{num_layer}.w;
    temp_w(1:model.classes,:) = temp_w(1:model.classes,:) * model.duplicate;
    for i = 1 : param.gibbs_iter
        % alternating gibbs
        % prop up
        hn = bsxfun(@plus, [batch_label, hn_1] * temp_w, model.layers{num_layer}.c);
        hn = 1 ./ (1 + exp(-hn));
        hn = single(hn > rand(size(hn)));

        % prop down
        hn_1 = bsxfun(@plus, hn * model.layers{num_layer}.w', model.layers{num_layer}.b);
        if ~exist('class','var')
            batch_label = exp(bsxfun(@minus, hn_1(:,1:model.classes), max(hn_1(:,1:model.classes), [], 2)));
            batch_label = bsxfun(@rdivide, batch_label, sum(batch_label, 2));
            batch_label  = mnrnd(1, batch_label);
        end
        hn_1 = 1 ./ ( 1 + exp(-hn_1(:,model.classes+1:end)));
        hn_1 = single(hn_1 > rand(size(hn_1)));
    end

    batch_data = reshape(hn_1, [batch_size, model.layers{num_layer-1}.layerSize]);

    for l = num_layer - 1 : -1 : 2
        if l == 2
            batch_data = reshape(batch_data, [batch_size, model.layers{l}.layerSize]);
            presigmoid = myConvolve(kConv_backward, batch_data, model.layers{l}.dw, model.layers{l}.stride, 'backward');
            presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
            batch_data = 1 ./ ( 1 + exp(-presigmoid));
        elseif strcmp(model.layers{l}.type, 'convolution')
            batch_data = reshape(batch_data, [batch_size, model.layers{l}.layerSize]);
            presigmoid = myConvolve(kConv_backward_c, batch_data, model.layers{l}.dw, model.layers{l}.stride, 'backward');
            presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
            batch_data = 1 ./ ( 1 + exp(-presigmoid));
        else
            batch_data = reshape(batch_data, [batch_size, model.layers{l}.layerSize]);
            presigmoid = bsxfun(@plus, ...
                batch_data * model.layers{l}.dw', model.layers{l}.b);
            batch_data = 1 ./ ( 1 + exp(-presigmoid) );
        end
    end
end

function [y] = sigmoid(x)
	y = 1 ./ (1 + exp(-x));
