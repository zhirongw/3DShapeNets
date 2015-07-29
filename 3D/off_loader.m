function offobj = off_loader(filename, theta, axis, stretch)
% read the off file. The function can also give some initial manipulations
% of the 3D data like rotation and stretch.
% offobj: a struct contains vertices and faces of 3D mesh models.
offobj = struct();
fid = fopen(filename, 'rb');
OFF_sign = fscanf(fid, '%c', 3);
assert(strcmp(OFF_sign, 'OFF') == 1);

info = fscanf(fid, '%d', 3);
offobj.vertices = reshape(fscanf(fid, '%f', info(1)*3), 3, info(1))';
offobj.faces = reshape(fscanf(fid, '%d', info(2)*4), 4, info(2))';
if ~isempty(find(offobj.faces(:,1) == 4, 1))
    fprintf('nononononono\n');
end

% do some translation and rotation
center = (max(offobj.vertices) + min(offobj.vertices)) / 2;
offobj.vertices = bsxfun(@minus, offobj.vertices, center);
if exist('axis', 'var')
    switch axis
        case 'x',
            offobj.vertices(:,1) = offobj.vertices(:,1) * stretch;
        case 'y',
            offobj.vertices(:,2) = offobj.vertices(:,2) * stretch;
        case 'z',
            offobj.vertices(:,3) = offobj.vertices(:,3) * stretch;
        otherwise,
            error('off_loader axis set wrong');
    end
end
theta = theta * pi / 180;
R = [cos(theta), -sin(theta), 0;
     sin(theta), cos(theta) , 0;
        0      ,    0       , 1];

offobj.vertices = offobj.vertices * R;

% These vertices to define faces should be offset by one to follow the matlab convention.
offobj.faces = offobj.faces(:,2:end) + 1; 

fclose(fid);