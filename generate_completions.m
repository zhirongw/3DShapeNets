function generate_completions(model, data_path, angle_inc)
% Generate completions for each off mesh models in data_path.
% In data_path, there should exist some off mesh files.
% rotate each mesh model by angle_inc.

addpath voxelization;
addpath 3D;
kernels;

volume_size = model.volume_size;
pad_size = model.pad_size;
data_size = volume_size + 2 * pad_size;

files = dir(data_path);
for i = 1 : length(files)
    if strcmp(files(i).name, '.') || strcmp(files(i).name, '..') || files(i).isdir == 1 || ~strcmp(files(i).name(end-2:end), 'off')
        continue;
    end
    filename = [data_path '/' files(i).name];
    for viewpoint = 1 : 360/angle_inc
        % random positions
        % obj_center = rand(3,1) - 0.5;
        % obj_center = 2 .* bsxfun(@rdivide, obj_center, sqrt(sum(obj_center.^2)));

        obj_center = [0,0,-2]'; % object coordinate in camera system
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
        gridDists = TSDF(depth, K, obj_center, R, trans, volume_size * mult, pad_size * mult, [], crop);

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
        % 3D Shapenets inference
        [completed_samples, ~] = rec_completion_test(model, batch_data, mask, 0, sample_param);
        
        save([filename(1:end-4) '_completions'], 'completed_samples');
    end
end