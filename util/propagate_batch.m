function batch = propagate_batch(model, batch, layer)
% propagate the batch from the bottom to the layer specified.
% also works for layer == 2.

global kConv_forward2;
global kConv_forward_c;

for l = 2 : layer - 1
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
    else
        batch_size = size(batch,1);
        batch = reshape(batch, batch_size, []);
        hidden_presigmoid = bsxfun(@plus, ...
            batch * model.layers{l}.w, model.layers{l}.c);
        batch = 1 ./ ( 1 + exp(- hidden_presigmoid) );
    end
end
