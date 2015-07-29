function batch = read_batch(model, file_list, translation)
% load the input batch from file(pre-computed mat files)
% translation: 1 for translating the data by small move and 0 for nothing.
% Hopefully, introducing translation on data would train a better model.

batch_size = length(file_list);
batch = zeros([batch_size, model.layers{1}.layerSize], 'int8');
for i = 1 : batch_size
    load(file_list(i).filename);
    if translation
        orient = randi(7);
        move = 2;
        shifted = zeros(size(instance), 'int8');
        switch orient
            case 1, % 'z'
                shifted(:,:,1:end-move) = instance(:,:,move+1:end);
            case 2, % 'z'
                shifted(:,:,move+1:end) = instance(:,:,1:end-move);
            case 3, % 'y'
                shifted(:,1:end-move,:) = instance(:,move+1:end,:);
            case 4, % 'y'
                shifted(:,move+1:end,:) = instance(:,1:end-move,:);
            case 5, % 'x'
                shifted(1:end-move,:,:) = instance(move+1:end,:,:);
            case 6, % 'x'
                shifted(move+1:end,:,:) = instance(1:end-move,:,:);
            case 7, % 'none'
                shifted = instance;
        end
    else
        shifted = instance;
    end
    batch(i,:,:,:) = shifted;
end

batch = single(batch);
