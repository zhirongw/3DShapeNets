function acc = TSDF_recognition_test(model, data_path, obj_position, angle_inc)
% view-based recognition test
% data_path: input off mesh root path for all classes

addpath 3D
addpath voxelization

classes = model.classnames;
volume_size = model.volume_size;
pad_size = model.pad_size;
data_size = volume_size + 2 * pad_size;

test_label = zeros(1);
predicted_label = zeros(1);

index = 0;
for c = 1 : length(classes)
    fprintf('testing on %s class\n', classes{c});
    category_path = [data_path '/' classes{c} '/test'];
    files = dir(category_path);
    
    for i = 1 : length(files)
        if strcmp(files(i).name, '.') || strcmp(files(i).name, '..') || files(i).isdir == 1 || ~strcmp(files(i).name(end-2:end), 'off')
            continue;
        end
        filename = [category_path '/' files(i).name];
        for viewpoint = 1 : 360/angle_inc
            obj_center = obj_position'; % object coordinate world coordinate
            camera_center = [0,0,0]'; camera_direction = [0,0,1]';

            axis = cross(camera_direction, obj_center - camera_center);
            if all(axis == 0)
               axis = cross(camera_direction, [1,0,0]'); 
            end
            angle = atan2(norm(cross(camera_direction, obj_center - camera_center)),dot(camera_direction, obj_center - camera_center));
            axis_angle = axis / norm(axis) * (-angle);
            R_o = AngleAxis2RotationMatrix(axis_angle); trans_o = [0,0,0]';
    
            [depth_new, K, crop] = off2im(filename, 1, (viewpoint - 1) * angle_inc * pi / 180, R_o, obj_center(1), obj_center(2), obj_center(3), [1;1;1], 0, 0);
            depth{1} = depth_new; R{1} = R_o; trans{1} = trans_o; mult = 5;
            gridDists = TSDF(depth, K, obj_center, R, trans, volume_size * mult, pad_size * mult, crop);
            
            gridDists = cubicle2col(gridDists, mult);
            surface_num = sum((gridDists < 1 & gridDists > -1),1);
            out_num = sum((gridDists == 1),1);
            in_num = sum((gridDists == -1),1);
            
            sur_index = (surface_num > 0 & in_num > 0 & out_num > 0) | surface_num > mult^2;
            out_index = (out_num >= in_num) & ~sur_index;
            
            gridDists = ones([data_size, data_size, data_size]);
            gridDists = -1 * gridDists;
            gridDists(sur_index) = 1;
            gridDists(out_index) = 0;
            gridDists = permute(gridDists, [3,1,2]);
            
            sample_param = [];
            sample_param.epochs = 30;
            sample_param.nparticles = 9;
            sample_param.gibbs_iter = 1;
            sample_param.earlyStop = true;

            batch_data = repmat(permute(gridDists,[4,1,2,3]),sample_param.nparticles,1); % running n chains altogether
            mask = batch_data < 0;

            [~, this_label] = rec_completion_test(model, batch_data, mask, 0, sample_param);
            
            index = index + 1;
            test_label(index, 1) = c;
            [~, predicted_label(index, 1)] = max(mean(this_label, 1));
        end
    end
end

for c = 1 : length(classes)
    num_class = sum(test_label == c);
    num_class_correct = sum(test_label == c & predicted_label == c);
    fprintf('class %s: correct %d of %d, acc: %f\n',...
        classes{c}, num_class_correct, num_class, num_class_correct / num_class);
end
acc = sum(predicted_label == test_label) ./ size(test_label,1);
fprintf('total %d, acc: %f\n', size(test_label, 1), acc);
