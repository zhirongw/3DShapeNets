function model = merge_model(model)
% get initialization for finetuning from pretrained model required by wake-sleep finetuning.

num_layer = length(model.layers);
for l = 2 : num_layer - 1
	model.layers{l}.uw = model.layers{l}.w;
	model.layers{l}.dw = model.layers{l}.w;
	model.layers{l} = rmfield(model.layers{l},'w');
    if isfield(model.layers{l}, 'grdw')
        model.layers{l} = rmfield(model.layers{l},'grdw');
        model.layers{l} = rmfield(model.layers{l},'grdb');
        model.layers{l} = rmfield(model.layers{l},'grdc');
    end
    model.layers{l}.grdb = zeros(size(model.layers{l}.b), 'single');
    model.layers{l}.grdc = zeros(size(model.layers{l}.c), 'single');
    model.layers{l}.grduw = zeros(size(model.layers{l}.uw), 'single');
	model.layers{l}.grddw = zeros(size(model.layers{l}.dw), 'single');
end

model.layers{num_layer}.grdb = zeros(size(model.layers{num_layer}.b), 'single');
model.layers{num_layer}.grdc = zeros(size(model.layers{num_layer}.c), 'single');
model.layers{num_layer}.grdw = zeros(size(model.layers{num_layer}.w), 'single');