function [model] = bp_update(model, param)
% backprop update used for discriminative finetuning

momentum = param.momentum;
lr = param.lr;
wd = param.weight_decay;
numLayer = model.numLayer;

for l = 2 : model.numLayer - 1
    model.layers{l}.histw = momentum * model.layers{l}.histw + lr * (model.layers{l}.grdw + wd * model.layers{l}.w);
    model.layers{l}.histc = momentum * model.layers{l}.histc + lr * model.layers{l}.grdc;
    model.layers{l}.w = model.layers{l}.w - (model.layers{l}.histw);
    model.layers{l}.c = model.layers{l}.c - (model.layers{l}.histc);
end

model.layers{numLayer}.histw = momentum * model.layers{numLayer}.histw + lr / 20 * (model.layers{numLayer}.grdw + wd * model.layers{numLayer}.w);
model.layers{numLayer}.histc = momentum * model.layers{numLayer}.histc + lr / 20 * model.layers{numLayer}.grdc;
model.layers{numLayer}.w = model.layers{numLayer}.w - (model.layers{numLayer}.histw);
model.layers{numLayer}.c = model.layers{numLayer}.c - (model.layers{numLayer}.histc);

% monitor the update magnitude for debugging
%{
for l = 2 : model.numLayer
    wsum = mean(abs(model.layers{l}.w(:))); wdsum = mean(abs(model.layers{l}.histw(:)));
    csum = mean(abs(model.layers{l}.c(:))); cdsum = mean(abs(model.layers{l}.histc(:)));
    fprintf('layer: %d\n', l);
    fprintf('w: %f, grdw: %f\n',wsum, wdsum);
    fprintf('c: %f, grdc: %f\n',csum, cdsum);
end
%}