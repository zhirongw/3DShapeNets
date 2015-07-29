function model = initialize_cdbn(param)
% initialize the model given the architecture.

% get the parameter for initializing the network
network = param.network;
numLayer = length(network);

% initialize the model
model = [];
model.numLayer = numLayer;
model.classnames = param.classnames;
model.classes = length(param.classnames);
model.layers = cell(numLayer, 1);
model.validation = param.validation;
model.duplicate = param.duplicate;

model.volume_size = param.volume_size;
model.pad_size = param.pad_size;
model.data_size = param.data_size;
model.data_path = param.data_path;

model.layers{1}.type = 'input';
model.layers{1}.layerSize = param.data_size;
for l = 2 : numLayer
    if strcmp(network{l}.type , 'convolution')
        model.layers{l}.type = 'convolution';
        model.layers{l}.actFun = network{l}.actFun;
        model.layers{l}.stride = network{l}.stride;
        
        preLayerSize = model.layers{l-1}.layerSize;
        thisMapSize = (preLayerSize(1:3) - network{l}.kernelSize) / model.layers{l}.stride + 1;
        assert(all(thisMapSize == floor(thisMapSize)), 'hidden layer are not integers');
        
        model.layers{l}.layerSize = [thisMapSize network{l}.outputMaps];
        model.layers{l}.kernelSize = [network{l}.kernelSize, network{l}.kernelSize, network{l}.kernelSize];

        model.layers{l}.w = rand([network{l}.outputMaps, network{l}.kernelSize, network{l}.kernelSize, network{l}.kernelSize, model.layers{l-1}.layerSize(4)], 'single');
        model.layers{l}.w = (model.layers{l}.w - 0.5) * 2 * sqrt( 6 / (network{l}.kernelSize.^3 * (network{l}.outputMaps + model.layers{l-1}.layerSize(4))));
        model.layers{l}.c  = zeros([network{l}.outputMaps, 1],'single'); 
        model.layers{l}.b = zeros([model.layers{l-1}.layerSize, 1],'single');
		
        model.layers{l}.grdw = zeros(size(model.layers{l}.w),'single');
        model.layers{l}.grdc = zeros(size(model.layers{l}.c),'single');
        model.layers{l}.grdb = zeros(size(model.layers{l}.b),'single');
        
    elseif strcmp(network{l}.type , 'fullconnected') && l < numLayer
        model.layers{l}.type = 'fullconnected';
        model.layers{l}.actFun = network{l}.actFun;
        model.layers{l}.layerSize = [network{l}.size 1];
		
        model.layers{l}.w = rand([prod(model.layers{l-1}.layerSize), prod(model.layers{l}.layerSize)], 'single');
        model.layers{l}.w = (model.layers{l}.w - 0.5) * 2 * sqrt( 6 / (prod(model.layers{l}.layerSize) + prod(model.layers{l-1}.layerSize)));
        model.layers{l}.c = zeros([1, model.layers{l}.layerSize], 'single');
        model.layers{l}.b = zeros([1, prod(model.layers{l-1}.layerSize)],'single');
        
        model.layers{l}.grdw = zeros(size(model.layers{l}.w), 'single');
        model.layers{l}.grdc = zeros(size(model.layers{l}.c), 'single');
        model.layers{l}.grdb = zeros(size(model.layers{l}.b), 'single');
    else
        model.layers{l}.type = 'fullconnected';
        model.layers{l}.actFun = network{l}.actFun;
        model.layers{l}.layerSize = [network{l}.size 1];
		
        model.layers{l}.w = rand(prod(model.layers{l-1}.layerSize) + param.classes, prod(model.layers{l}.layerSize), 'single');
        model.layers{l}.w = (model.layers{l}.w - 0.5) * 2 * sqrt( 6 / (prod(model.layers{l}.layerSize) + prod(model.layers{l-1}.layerSize)));
        model.layers{l}.c = zeros([1, model.layers{l}.layerSize], 'single');
        model.layers{l}.b = zeros([1, prod(model.layers{l-1}.layerSize) + param.classes],'single');
        
        model.layers{l}.grdw = zeros(size(model.layers{l}.w),'single');
        model.layers{l}.grdc = zeros(size(model.layers{l}.c),'single');
        model.layers{l}.grdb = zeros(size(model.layers{l}.b),'single');
    end
end
