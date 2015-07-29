function [model, activation] = bp_forward(model, batch)
% backprop feed forward used for discriminative finetuning

global kConv_forward2;
global kConv_forward_c;

activation = cell(model.numLayer, 1);
for l = 2 : model.numLayer
    activation{l-1} = batch;
    if l == 2
        stride = model.layers{l}.stride;
        hidden_presigmoid = myConvolve2(kConv_forward2, batch, model.layers{l}.w, stride, 'forward');
        hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
        batch = 1 ./ (1 + exp(-hidden_presigmoid));
    elseif strcmp(model.layers{l}.type, 'convolution')
        stride = model.layers{l}.stride;
        hidden_presigmoid = myConvolve(kConv_forward_c, batch, model.layers{l}.w, stride, 'forward');
        hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
        batch = 1 ./ (1 + exp(-hidden_presigmoid));
    elseif strcmp(model.layers{l}.type, 'fullconnected') && l < model.numLayer
        batch_size = size(batch,1);
        batch = reshape(batch, batch_size, []);
        hidden_presigmoid = bsxfun(@plus, double(batch) * double(model.layers{l}.w), double(model.layers{l}.c));
        batch = 1 ./ (1 + exp(-hidden_presigmoid));
    else
        batch_size = size(batch,1);
        batch = reshape(batch, batch_size, []);
        temp = bsxfun(@plus, batch * model.layers{l}.w, model.layers{l}.c);
        energy = exp( bsxfun (@minus, temp, max(temp, [], 2)));
        batch = bsxfun(@rdivide, energy, sum(energy,2));
    end
end
activation{end} = batch;
