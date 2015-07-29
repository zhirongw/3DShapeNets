function [model, loss] = bp_backward(model, activation, label)
% backprop backward used for discriminative finetuning

 global kConv_backward kConv_backward_c;
 global kConv_weight kConv_weight_c;

numLayer = model.numLayer;
loss = - mean( log(activation{numLayer}(label == 1) ) );

batch_size = size(label,1);
error = cell(numLayer, 1);
error{numLayer} = - ( double(label) - activation{numLayer} );
for l = model.numLayer-1 : -1 : 2
    if l == 1
        error{l} = myConvolve(kConv_backward, error{l+1}, model.layers{l+1}.w, model.layers{l+1}.stride, 'backward');
    elseif strcmp(model.layers{l+1}.type, 'convolution')
        error{l} = myConvolve(kConv_backward_c, error{l+1}, model.layers{l+1}.w, model.layers{l+1}.stride, 'backward');
    else
		error{l} = double(error{l+1}) * double(model.layers{l+1}.w');
        if strcmp(model.layers{l}.type, 'convolution')
            error{l} = reshape(error{l}, [batch_size, model.layers{l}.layerSize]);
        end
    end
    error{l} = error{l} .* double(activation{l}) .* double(( 1 - activation{l} ));
end

% Compute the gradients for each layer
for l = 2 : numLayer
    if l == 2
        model.layers{l}.grdw = myConvolve(kConv_weight, activation{l-1}, error{l}, model.layers{l}.stride, 'weight')*size(error{l},2)^3;
        model.layers{l}.grdc = sum(reshape(error{l}, [], model.layers{l}.layerSize(4)),1)' ./ batch_size; 
    elseif strcmp(model.layers{l}.type, 'convolution')
        model.layers{l}.grdw = myConvolve(kConv_weight_c, activation{l-1}, error{l}, model.layers{l}.stride, 'weight')*size(error{l},2)^3;
        model.layers{l}.grdc = sum(reshape(error{l}, [], model.layers{l}.layerSize(4)),1)' ./ batch_size; 
    else
        activation{l-1} = reshape(activation{l-1}, batch_size, []);
        model.layers{l}.grdw = double(activation{l-1}') * error{l}./ batch_size;
        model.layers{l}.grdc = mean(error{l}, 1);
    end
end
