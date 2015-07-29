function [depth,K,crop] = off2im(offfile, ratio, xzRot, Rtilt, objx,objy, objz, modelsize,addfloor,enlargefloor)
% Render a depth map from a 3D mesh model provided by Shuran Song.
% calls RenderMex

% offfile: off filename
% ratio: set it 1
% xzRot: rotate angle of the 3D mesh model
% Rtilt: tilt angle of the 3D mesh model
% objx, objy, objz: object position in the world coordinate
% modelsize: set it 1
% addfloor: whether to add a floor at the bottom
% enlargefloor: whether to enlarge the floor
% depth: returned depth map
% K: camera intrinsic
% crop: just always let it be 1

if ~exist('xzRot', 'var')
    xzRot = rand() * pi*2;
end
if ~exist('objz', 'var')
    objz = 2 + rand() * 5;   % 2 to 5
end
if ~exist('objx', 'var')
    objx = (rand()*0.5-0.25) .* objz;
end
if ~exist('ratio', 'var')
    ratio = 1;
end
if ~exist('objy', 'var')
    objy = 1.3;
end
if ~exist('modelsize', 'var')
    modelsize = [1;1;1]*rand()*1.5+0.5;
end
if ~exist('addfloor','var')
    addfloor =1;
end
    

%% Camera Paramter
imw = 640 * ratio; 
imh = 480 * ratio;
fx_rgb = 5.19e+02 * ratio;
fy_rgb = 5.19e+02 * ratio;
cx_rgb = imw/2;
cy_rgb = imh/2;
K=[fx_rgb 0 cx_rgb; 0 fy_rgb cy_rgb; 0 0 1];

C = [0;0;0];

z_near = 0.3;
z_far_ratio = 1.2;
Ryzswi = [1, 0, 0; 0, 0, 1; 0, 1, 0];

%%
offobj = offLoader(offfile);
offobj.vmat = Ryzswi * offobj.vmat;
Robj = genRotMat(xzRot);
P = K * Rtilt * [eye(3), -C];

vmat = scalePoints(Robj * offobj.vmat, [objx;objy;objz], modelsize);
if addfloor
    minv = min(vmat, [], 2);
    maxv = max(vmat, [], 2);
    %make the floor larger
    v1=[minv(1)-enlargefloor;minv(2);minv(3)-enlargefloor];
    v2=[minv(1)-enlargefloor;minv(2);maxv(3)+enlargefloor];
    v3=[maxv(1)+enlargefloor;minv(2);maxv(3)+enlargefloor];
    v4=[maxv(1)+enlargefloor;minv(2);minv(3)-enlargefloor];
    vmat = [vmat,v1,v2,v3,v4];
    fmat = [offobj.fmat,[length(vmat)-4;length(vmat)-1;length(vmat)-2;length(vmat)-4]...
               ,[length(vmat)-3-1;length(vmat)-1-1;length(vmat)-2-1;length(vmat)-3-1]];
    offobj.fmat=fmat;
end

result = RenderMex(P, imw, imh, [vmat(1,:);vmat(2,:);vmat(3,:)], uint32(offobj.fmat))';
depth = z_near./(1-double(result)/2^32);
maxDepth = 10;
cropmask = (depth < z_near) | (depth > z_far_ratio * maxDepth);
depth(cropmask) = NaN;
crop =[1,1];
end

function offobj = offLoader(filename)

offobj = struct();
fid = fopen(filename, 'rb');
OFF_sign = fscanf(fid, '%c', 3);
assert(strcmp(OFF_sign, 'OFF') == 1);

words = fscanf(fid, '%d', 3);
nV = words(1); nF = words(2); nE = words(3);
offobj.vmat = reshape(fscanf(fid, '%f', nV*3), 3, nV);
fstr = textscan(fid, '%s', nF, 'delimiter', '\n', 'MultipleDelimsAsOne', 1);
tfmat = zeros(nF*2,4);
nf3 = 0;
for i=1:nF
    words = sscanf(fstr{1}{i}, '%d');
    if words(1) == 3
        nf3 = nf3 + 1;
        tfmat(nf3, :) = words([4,2,3,4]);
    elseif words(1) == 4
        nf3 = nf3 + 1;
        tfmat(nf3, :) = words([4,2,3,4]);
        nf3 = nf3 + 1;
        tfmat(nf3, :) = words([5,2,4,5]);
    else
        error('size of face is not 3 or 4');
    end
end
offobj.fmat = tfmat(1:nf3, :)';
fclose(fid);
end

function R = genRotMat(theta)

R = [cos(theta),0, -sin(theta);
    0,  1, 0;
    sin(theta),0, cos(theta)];

end

function coornew = scalePoints(coor, center, size)
% parameters:
%   coor: 3*n coordinates of n points
%   center: 3*1 the center of new point cloud
%   size: 3*1 the size of new point cloud
minv = min(coor, [], 2);
maxv = max(coor, [], 2);
oldCenter = (minv+maxv)/2;
oldSize = maxv - minv;
scale = min(size./ oldSize);
coornew = bsxfun(@plus, scale * coor, center-scale*oldCenter);

end
