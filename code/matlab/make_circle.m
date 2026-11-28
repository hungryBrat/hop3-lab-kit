function P = make_circle(centre, R, n, normal)
% n points of a closed circle (3xn mm), point 1 at angle 0, last point one step short of the start.
if nargin < 4 || isempty(normal), normal = [0 0 1]; end
c = centre(:); nz = normal(:)/norm(normal);
u = [1;0;0] - nz*nz(1);
if norm(u) < 1e-9, u = [0;1;0] - nz*nz(2); end
u = u/norm(u); v = cross(nz, u);
phi = 2*pi*(0:n-1)/n;
P = c + R*(u*cos(phi) + v*sin(phi));
end
