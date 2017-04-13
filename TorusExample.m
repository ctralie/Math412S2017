%Programmer: Chris Tralie
%Purpose: To show how the Matlab ripser wrapper works on a 2-torus

[theta, phi] = meshgrid(linspace(0, 2*pi, 25), linspace(0, 2*pi, 12));
R1 = 2;
R2 = 1;

x = (R1 + R2.*cos(phi)) .* cos(theta);
y = (R1 + R2.*cos(phi)) .* sin(theta);
z = R2.*sin(phi);
X = zeros(length(theta(:)), 3);
X(:, 1) = x(:);
X(:, 2) = y(:);
X(:, 3) = z(:);


PDs = RipsFiltrationPC(X, 2);
subplot(121);
plot3(X(:, 1), X(:, 2), X(:, 3), '.');
axis equal;
subplot(122);
plotpersistencediagrams(PDs);