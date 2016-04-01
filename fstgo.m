% Transform the dummy image from PBC to CGC system. 
x=imresize(imread('C:\Users\Gaurav\Desktop\bpcs_project\cameraman.bmp'),[256 256]);
R=imresize(imread('C:\Users\Gaurav\Desktop\bpcs_project\Wc.png'),[8 8]);
W=imresize(imread('C:\Users\Gaurav\Desktop\lena-binary.jpg'),[64 64]);
A=zeros(256,256);
for i=1:256
A(i,1)=x(i,1);
end 
for i=1:256
for j=2:256 
    A(i,j)=bitxor(x(i,j-1),x(i,j)); 
end 
end 
subplot(1,2,1);
imshow(x);
title('Original Image');
subplot(1,2,2);
imshow(A);
title('Image in CGC System');

%Segment each bit-plane of the dummy image into informative and noise-like regions by using a threshold value. A typical value is ?0 = 0.3. 
%Bit plane slicing
B=bitget(A,1);
C=bitget(A,2);
D=bitget(A,3);
E=bitget(A,4);
F=bitget(A,5);
G=bitget(A,6);
H=bitget(A,7);
I=bitget(A,8);

figure,
subplot(2,4,1);imshow(logical(B));title('Bitplane 1');     %Show Bitplane 1 
subplot(2,4,2);imshow(logical(C));title('Bitplane 2');     %Show Bitplane 2
subplot(2,4,3);imshow(logical(D));title('Bitplane 3');     %Show Bitplane 3
subplot(2,4,4);imshow(logical(E));title('Bitplane 4');     %Show Bitplane 4
subplot(2,4,5);imshow(logical(F));title('Bitplane 5');     %Show Bitplane 5
subplot(2,4,6);imshow(logical(G));title('Bitplane 6');     %Show Bitplane 6
subplot(2,4,7);imshow(logical(H));title('Bitplane 7');     %Show Bitplane 7
subplot(2,4,8);imshow(logical(I));title('Bitplane 8');     %Show Bitplane 8

[r c]=size(B);
bs=8; % Block Size (8x8)
Block=zeros(8,8);
% Dividing the image into 8x8 Blocks
kk=0;
for i=1:(r/bs)
for j=1:(c/bs)
    Block(:,:,kk+j)=B((bs*(i-1)+1:bs*(i-1)+bs),(bs*(j-1)+1:bs*(j-1)+bs));
end
kk=kk+(r/bs);
end
c7=0;c8=0;c9=0;c10=0;c11=0;c12=0;c13=0;c14=0;
z=zeros(1024,1);
z1=zeros(1024,1);
y=zeros(8,8);
k=1;
for f=1:1024
    y=Block(:,:,f);   
    for h=1:7           % Complexity of entire image
        for l=1:7
            v=bitxor(y(l,h),y(l+1,h));
            v1=bitxor(y(h,l),y(h,l+1));
            if v==1
                c7=c7+1;
            end    
            if  v1==1
                c8=c8+1;    
            end
        end
    end
    for h=1:7           % Complexity of image border
            v2=bitxor(y(1,h),y(1,h+1));
            v3=bitxor(y(8,h),y(8,h+1));
            v4=bitxor(y(h,8),y(h+1,8));
            v5=bitxor(y(h,1),y(h+1,1));
            if  v2==1
                c11=c11+1;
            end
            if  v3==1
                c10=c10+1;
            end 
            if v4==1
                c12=c12+1;
            end
            if v5==1
                c13=c13+1;
            end
    end
    c9=c7+c8; c14=c10+c11+c12+c13;
    z(k)=c9;    z1(k)=c14; k=k+1;
    c7=0;c8=0;c10=0;c11=0;c12=0;c13=0;c9=0;c14=0;
    v=0;v1=0;v2=0;v3=0;v4=0;v5=0;
end
for i=1:1024
  z(i)=z1(i)/z(i); 
end

[r c]=size(C);
Block=zeros(8,8);
% Dividing the image into 8x8 Blocks
kk=0;
for i=1:(r/bs)
for j=1:(c/bs)
    Block(:,:,kk+j)=C((bs*(i-1)+1:bs*(i-1)+bs),(bs*(j-1)+1:bs*(j-1)+bs));
end
kk=kk+(r/bs);
end
c7=0;c8=0;c9=0;c10=0;c11=0;c12=0;c13=0;c14=0;
z2=zeros(1024,1);
z3=zeros(1024,1);
y=zeros(8,8);
k=1;
for f=1:1024
    y=Block(:,:,f);   
    for h=1:7           % Complexity of entire image
        for l=1:7
            v=bitxor(y(l,h),y(l+1,h));
            v1=bitxor(y(h,l),y(h,l+1));
            if v==1
                c7=c7+1;
            end    
            if  v1==1
                c8=c8+1;    
            end
        end
    end
    for h=1:7           % Complexity of image border
            v2=bitxor(y(1,h),y(1,h+1));
            v3=bitxor(y(8,h),y(8,h+1));
            v4=bitxor(y(h,8),y(h+1,8));
            v5=bitxor(y(h,1),y(h+1,1));
            if  v2==1
                c11=c11+1;
            end
            if  v3==1
                c10=c10+1;
            end 
            if v4==1
                c12=c12+1;
            end
            if v5==1
                c13=c13+1;
            end
    end
    c9=c7+c8; c14=c10+c11+c12+c13;
    z2(k)=c9;    z3(k)=c14; k=k+1;
    c7=0;c8=0;c10=0;c11=0;c12=0;c13=0;c9=0;c14=0;
    v=0;v1=0;v2=0;v3=0;v4=0;v5=0;
end
for i=1:1024
  z2(i)=z3(i)/z2(i); 
end

[r c]=size(D);
Block=zeros(8,8);
% Dividing the image into 8x8 Blocks
kk=0;
for i=1:(r/bs)
for j=1:(c/bs)
    Block(:,:,kk+j)=D((bs*(i-1)+1:bs*(i-1)+bs),(bs*(j-1)+1:bs*(j-1)+bs));
end
kk=kk+(r/bs);
end
c7=0;c8=0;c9=0;c10=0;c11=0;c12=0;c13=0;c14=0;
z4=zeros(1024,1);
z5=zeros(1024,1);
y=zeros(8,8);
k=1;
for f=1:1024
    y=Block(:,:,f);   
    for h=1:7           % Complexity of entire image
        for l=1:7
            v=bitxor(y(l,h),y(l+1,h));
            v1=bitxor(y(h,l),y(h,l+1));
            if v==1
                c7=c7+1;
            end    
            if  v1==1
                c8=c8+1;    
            end
        end
    end
    for h=1:7           % Complexity of image border
            v2=bitxor(y(1,h),y(1,h+1));
            v3=bitxor(y(8,h),y(8,h+1));
            v4=bitxor(y(h,8),y(h+1,8));
            v5=bitxor(y(h,1),y(h+1,1));
            if  v2==1
                c11=c11+1;
            end
            if  v3==1
                c10=c10+1;
            end 
            if v4==1
                c12=c12+1;
            end
            if v5==1
                c13=c13+1;
            end
    end
    c9=c7+c8; c14=c10+c11+c12+c13;
    z4(k)=c9;    z5(k)=c14; k=k+1;
    c7=0;c8=0;c10=0;c11=0;c12=0;c13=0;c9=0;c14=0;
    v=0;v1=0;v2=0;v3=0;v4=0;v5=0;
end
for i=1:1024
  z4(i)=z5(i)/z4(i); 
  if isnan(z4(i)) || isinf(z4(i))
      z4(i)=0;
  end    
end

[r c]=size(E);
Block=zeros(8,8);
% Dividing the image into 8x8 Blocks
kk=0;
for i=1:(r/bs)
for j=1:(c/bs)
    Block(:,:,kk+j)=E((bs*(i-1)+1:bs*(i-1)+bs),(bs*(j-1)+1:bs*(j-1)+bs));
end
kk=kk+(r/bs);
end
c7=0;c8=0;c9=0;c10=0;c11=0;c12=0;c13=0;c14=0;
z6=zeros(1024,1);
z7=zeros(1024,1);
y=zeros(8,8);
k=1;
for f=1:1024
    y=Block(:,:,f);   
    for h=1:7           % Complexity of entire image
        for l=1:7
            v=bitxor(y(l,h),y(l+1,h));
            v1=bitxor(y(h,l),y(h,l+1));
            if v==1
                c7=c7+1;
            end    
            if  v1==1
                c8=c8+1;    
            end
        end
    end
    for h=1:7           % Complexity of image border
            v2=bitxor(y(1,h),y(1,h+1));
            v3=bitxor(y(8,h),y(8,h+1));
            v4=bitxor(y(h,8),y(h+1,8));
            v5=bitxor(y(h,1),y(h+1,1));
            if  v2==1
                c11=c11+1;
            end
            if  v3==1
                c10=c10+1;
            end 
            if v4==1
                c12=c12+1;
            end
            if v5==1
                c13=c13+1;
            end
    end
    c9=c7+c8; c14=c10+c11+c12+c13;
    z6(k)=c9;    z7(k)=c14; k=k+1;
    c7=0;c8=0;c10=0;c11=0;c12=0;c13=0;c9=0;c14=0;
    v=0;v1=0;v2=0;v3=0;v4=0;v5=0;
end
for i=1:1024
  z6(i)=z7(i)/z6(i);
  if isnan(z6(i)) || isinf(z6(i))
      z6(i)=0;
  end
end

[r c]=size(F);
Block=zeros(8,8);
% Dividing the image into 8x8 Blocks
kk=0;
for i=1:(r/bs)
for j=1:(c/bs)
    Block(:,:,kk+j)=F((bs*(i-1)+1:bs*(i-1)+bs),(bs*(j-1)+1:bs*(j-1)+bs));
end
kk=kk+(r/bs);
end
c7=0;c8=0;c9=0;c10=0;c11=0;c12=0;c13=0;c14=0;
z8=zeros(1024,1);
z9=zeros(1024,1);
y=zeros(8,8);
k=1;
for f=1:1024
    y=Block(:,:,f);   
    for h=1:7           % Complexity of entire image
        for l=1:7
            v=bitxor(y(l,h),y(l+1,h));
            v1=bitxor(y(h,l),y(h,l+1));
            if v==1
                c7=c7+1;
            end    
            if  v1==1
                c8=c8+1;    
            end
        end
    end
    for h=1:7           % Complexity of image border
            v2=bitxor(y(1,h),y(1,h+1));
            v3=bitxor(y(8,h),y(8,h+1));
            v4=bitxor(y(h,8),y(h+1,8));
            v5=bitxor(y(h,1),y(h+1,1));
            if  v2==1
                c11=c11+1;
            end
            if  v3==1
                c10=c10+1;
            end 
            if v4==1
                c12=c12+1;
            end
            if v5==1
                c13=c13+1;
            end
    end
    c9=c7+c8; c14=c10+c11+c12+c13;
    z8(k)=c9;    z9(k)=c14; k=k+1;
    c7=0;c8=0;c10=0;c11=0;c12=0;c13=0;c9=0;c14=0;
    v=0;v1=0;v2=0;v3=0;v4=0;v5=0;
end
for i=1:1024
  z8(i)=z9(i)/z8(i); 
  if isnan(z8(i)) || isinf(z8(i))
      z8(i)=0;
  end    
end

[r c]=size(G);
Block=zeros(8,8);
% Dividing the image into 8x8 Blocks
kk=0;
for i=1:(r/bs)
for j=1:(c/bs)
    Block(:,:,kk+j)=G((bs*(i-1)+1:bs*(i-1)+bs),(bs*(j-1)+1:bs*(j-1)+bs));
end
kk=kk+(r/bs);
end
c7=0;c8=0;c9=0;c10=0;c11=0;c12=0;c13=0;c14=0;
z10=zeros(1024,1);
z11=zeros(1024,1);
y=zeros(8,8);
k=1;
for f=1:1024
    y=Block(:,:,f);   
    for h=1:7           % Complexity of entire image
        for l=1:7
            v=bitxor(y(l,h),y(l+1,h));
            v1=bitxor(y(h,l),y(h,l+1));
            if v==1
                c7=c7+1;
            end    
            if  v1==1
                c8=c8+1;    
            end
        end
    end
    for h=1:7           % Complexity of image border
            v2=bitxor(y(1,h),y(1,h+1));
            v3=bitxor(y(8,h),y(8,h+1));
            v4=bitxor(y(h,8),y(h+1,8));
            v5=bitxor(y(h,1),y(h+1,1));
            if  v2==1
                c11=c11+1;
            end
            if  v3==1
                c10=c10+1;
            end 
            if v4==1
                c12=c12+1;
            end
            if v5==1
                c13=c13+1;
            end
    end
    c9=c7+c8; c14=c10+c11+c12+c13;
    z10(k)=c9;    z11(k)=c14; k=k+1;
    c7=0;c8=0;c10=0;c11=0;c12=0;c13=0;c9=0;c14=0;
    v=0;v1=0;v2=0;v3=0;v4=0;v5=0;
end
for i=1:1024
  z10(i)=z11(i)/z10(i);
  if isnan(z10(i)) || isinf(z10(i))
      z10(i)=0;
  end    
end

[r c]=size(H);
bs=8; % Block Size (8x8)
Block=zeros(8,8);
% Dividing the image into 8x8 Blocks
kk=0;
for i=1:(r/bs)
for j=1:(c/bs)
    Block(:,:,kk+j)=H((bs*(i-1)+1:bs*(i-1)+bs),(bs*(j-1)+1:bs*(j-1)+bs));
end
kk=kk+(r/bs);
end
c7=0;c8=0;c9=0;c10=0;c11=0;c12=0;c13=0;c14=0;
z12=zeros(1024,1);
z13=zeros(1024,1);
y=zeros(8,8);
k=1;
for f=1:1024
    y=Block(:,:,f);   
    for h=1:7           % Complexity of entire image
        for l=1:7
            v=bitxor(y(l,h),y(l+1,h));
            v1=bitxor(y(h,l),y(h,l+1));
            if v==1
                c7=c7+1;
            end    
            if  v1==1
                c8=c8+1;    
            end
        end
    end
    for h=1:7           % Complexity of image border
            v2=bitxor(y(1,h),y(1,h+1));
            v3=bitxor(y(8,h),y(8,h+1));
            v4=bitxor(y(h,8),y(h+1,8));
            v5=bitxor(y(h,1),y(h+1,1));
            if  v2==1
                c11=c11+1;
            end
            if  v3==1
                c10=c10+1;
            end 
            if v4==1
                c12=c12+1;
            end
            if v5==1
                c13=c13+1;
            end
    end
    c9=c7+c8; c14=c10+c11+c12+c13;
    z12(k)=c9;    z13(k)=c14; k=k+1;
    c7=0;c8=0;c10=0;c11=0;c12=0;c13=0;c9=0;c14=0;
    v=0;v1=0;v2=0;v3=0;v4=0;v5=0;
end
for i=1:1024
  z12(i)=z13(i)/z12(i);
  if isnan(z12(i)) || isinf(z12(i))
      z12(i)=0;
  end    
end

[r c]=size(I);
Block1=zeros(8,8);
% Dividing the image into 8x8 Blocks
kk=0;
for i=1:(r/bs)
for j=1:(c/bs)
    Block1(:,:,kk+j)=I((bs*(i-1)+1:bs*(i-1)+bs),(bs*(j-1)+1:bs*(j-1)+bs));
end
kk=kk+(r/bs);
end
c7=0;c8=0;c9=0;c10=0;c11=0;c12=0;c13=0;c14=0;
z14=zeros(1024,1);
z15=zeros(1024,1);
y=zeros(8,8);
k=1;
for f=1:1024
    y=Block1(:,:,f);   
    for h=1:7           % Complexity of entire image
        for l=1:7
            v=bitxor(y(l,h),y(l+1,h));
            v1=bitxor(y(h,l),y(h,l+1));
            if v==1
                c7=c7+1;
            end    
            if  v1==1
                c8=c8+1;    
            end
        end
    end
    for h=1:7           % Complexity of image border
            v2=bitxor(y(1,h),y(1,h+1));
            v3=bitxor(y(8,h),y(8,h+1));
            v4=bitxor(y(h,8),y(h+1,8));
            v5=bitxor(y(h,1),y(h+1,1));
            if  v2==1
                c11=c11+1;
            end
            if  v3==1
                c10=c10+1;
            end 
            if v4==1
                c12=c12+1;
            end
            if v5==1
                c13=c13+1;
            end
    end
    c9=c7+c8; c14=c10+c11+c12+c13;
    z14(k)=c9;    z15(k)=c14; k=k+1;
    c7=0;c8=0;c10=0;c11=0;c12=0;c13=0;c9=0;c14=0;
    v=0;v1=0;v2=0;v3=0;v4=0;v5=0;
end 
for i=1:1024
  z14(i)=z15(i)/z14(i);
  if isnan(z14(i)) || isinf(z14(i))
      z14(i)=0;
  end
end

alpha=0.3;  c=0;
display('Bit Plane 1');
for i=1:1024            %Bit Plane 1
    if z(i)>alpha
        display(i);
        c=c+1;
    end
end
display(c);

c=0;
display('Bit Plane 2');
for i=1:1024            %Bit Plane 2
    if z2(i)>alpha
        display(i);
        c=c+1;
    end
end
display(c);

c=0;
display('Bit Plane 3');
for i=1:1024            %Bit Plane 3
    if z4(i)>alpha
        display(i);
        c=c+1;
    end
end
display(c);

c=0;
display('Bit Plane 4');
for i=1:1024            %Bit Plane 4
    if z6(i)>alpha
        display(i);
        c=c+1;
    end
end
display(c);

c=0;
display('Bit Plane 5');
for i=1:1024            %Bit Plane 5
    if z8(i)>alpha
        display(i);
        c=c+1;
    end
end
display(c);

c=0;
display('Bit Plane 6');
for i=1:1024            %Bit Plane 6
    if z10(i)>alpha
        display(i);
        c=c+1;
    end
end
display(c);

c=0;
display('Bit Plane 7');
for i=1:1024            %Bit Plane 7
    if z12(i)>alpha
        display(i);
        c=c+1;
    end
end
display(c);

c=0;
display('Bit Plane 8');
for i=1:1024            %Bit Plane 8
    if z13(i)>alpha
        display(i);
        c=c+1;
    end
end
display(c);

%Group the secret file ie. a binary image into a series of secret blocks.
% There would be 64 secret blocks of 8X8 size for a secret file of 64X64.
img=imresize(imread('C:\Users\Gaurav\Desktop\bpcs_project\binary.jpg'),[64 64]); 
figure,subplot(13,5,1);
imshow(img);
l=1;
k=2;
i=1;
j=1;
title('8X8 blocks of secret file');
for i=1:8:64
    for j=1:8:64
       J{l}=img(i:i+7,j:j+7);
       subplot(13,5,k);
       k=k+1;
       imshow(J{l});
       l=l+1;
    end
end


%Watermark file
figure,subplot(13,5,1);
imshow(W);
d=1;
k=2;
i=1;
j=1;
title('8X8 blocks of secret file');
for i=1:8:64
    for j=1:8:64
       P{d}=W(i:i+7,j:j+7);
       subplot(13,5,k);
       k=k+1;
       imshow(P{d});
       d=d+1;
    end
end


%Embedding secret file blocks in the noise blocks of bit planes.
s=1;
for i=1:1024
    if z(i)>alpha
        if s==65
            break;
        end    
        m1=floor((i-1)/32)*8+1;
        m2=m1+7;
        k3=mod((i-1),32);
        k1=floor(k3)*8+1; 
        k2=k1+7;
        B(m1:m2,k1:k2)=J{s};
        s=s+1;
    end
end

for i=1:1024
    if z2(i)>alpha
        if s==65
            break;
        end    
        m1=floor((i-1)/32)*8+1;
        m2=m1+7;
        k3=mod((i-1),32);
        k1=floor(k3)*8+1; 
        k2=k1+7;
        C(m1:m2,k1:k2)=J{s};
        s=s+1;
    end
end

for i=1:1024
    if z4(i)>alpha
        if s==65
            break;
        end    
        m1=floor((i-1)/32)*8+1;
        m2=m1+7;
        k3=mod((i-1),32);
        k1=floor(k3)*8+1; 
        k2=k1+7;
        D(m1:m2,k1:k2)=J{s};
        s=s+1;
    end
end

for i=1:1024
    if z6(i)>alpha
        if s==65
            break;
        end    
        m1=floor((i-1)/32)*8+1;
        m2=m1+7;
        k3=mod((i-1),32);
        k1=floor(k3)*8+1; 
        k2=k1+7;
        E(m1:m2,k1:k2)=J{s};
        s=s+1;
    end
end

for i=1:1024
    if z8(i)>alpha
        if s==65
            break;
        end    
        m1=floor((i-1)/32)*8+1;
        m2=m1+7;
        k3=mod((i-1),32);
        k1=floor(k3)*8+1; 
        k2=k1+7;
        F(m1:m2,k1:k2)=J{s};
        s=s+1;
    end
end

for i=1:1024
    if z10(i)>alpha
        if s==65
            break;
        end    
        m1=floor((i-1)/32)*8+1;
        m2=m1+7;
        k3=mod((i-1),32);
        k1=floor(k3)*8+1; 
        k2=k1+7;
        G(m1:m2,k1:k2)=J{s};
        s=s+1;
    end
end

for i=1:1024
    if z12(i)>alpha
        if s==65
            break;
        end    
        m1=floor((i-1)/32)*8+1;
        m2=m1+7;
        k3=mod((i-1),32);
        k1=floor(k3)*8+1; 
        k2=k1+7;
        H(m1:m2,k1:k2)=J{s};
        s=s+1;
    end
end

for i=1:1024
    if z14(i)>alpha
        if s==65
            break;
        end    
        m1=floor((i-1)/32)*8+1;
        m2=m1+7;
        k3=mod((i-1),32);
        k1=floor(k3)*8+1; 
        k2=k1+7;
        I(m1:m2,k1:k2)=J{s};
        s=s+1;
    end
end

d=1;
for i=1:1024
    if d==65
        break;
    end    
    if z(i)<0.3
        m1=floor((i-1)/32)*8+1;
        m2=m1+7;
        k3=mod((i-1),32);
        k1=floor(k3)*8+1; 
        k2=k1+7;
        B(m1:m2,k1:k2)=P{d};
        d=d+1;
    end
end    
        

%conmap=zeros(1024,1);
%display('Computing conjugate of all informative blocks of Bit Plane 1');
%for i=1:1024    %Bit Plane 1
 %   if z(i)<alpha
  %      conmap(i)=1;
   %     m1=floor((i-1)/32)*8+1;
    %    m2=m1+7;
     %   k3=mod((i-1),32);
      %  k1=floor(k3)*8+1; 
       % k2=k1+7;
        %u=1;
      %for m=m1:m2
       % v=1;  
      %for k=k1:k2
      %B(k,m)=bitxor(B(k,m),R(u,v)); 
      %v=v+1;
      %end 
      %u=u+1;
      %end 
    %end 
%end   

%***********************ENTROPY calculation*********************

%The entropy is calculated of each 8x8 block in every bit plane.
%Then we will find out the maximum entropy from each bit plane.
%And get the maximum entropy from the maximums.


% J{1}=B;
% J{2}=C;
% J{3}=D;
% J{4}=E;
% J{5}=F;
% J{6}=G;
% J{7}=H;
% J{8}=I;
% 
% row=size(x,1)/8;
% col=size(x,2)/8;
% 
% maxresult=zeros(1,8)
% c=1;
% for i=1:8
%     curr_plane=J{i};
%     blocks=mat2cell(curr_plane,ones(1,row)*8,ones(1,col)*8); %Dividing current plane into 8x8 block 
%     for bi=1:size(blocks,1)
%       for bj=1:size(blocks,2)
%          my_8x8_block=blocks{bi,bj};
%          a(c)=entropy(nanmean(my_8x8_block)); %Calculating entropy of block
%          Q{c}=a(c);
%          c=c+1;
%       end
%     end 
%     
%     % Extracting the maximum entropy block for each bit plane.
%      c=1;
%      maxim=0;
%     for bi=1:size(blocks,1)
%        for bj=1:size(blocks,2)
%            if(a(c)>maxim)
%                maxim=a(c);
%                blocknumber=c;
%            end
%            c=c+1;  
%        end
%     end
%     maxresult(i)=maxim;
%     block(i)=blocknumber;
%     P{i}=Q;
%     c=1;
% end
% 
% %Extracting the maximum entropy and its corresponding bit plane from the maximums. 
% 
% maxi=0;
% for i=1:8
%    if(maxresult(i)>maxi)
%        maxi=maxresult(i)
%        corresponding_block=block(i)
%        curr_plane=i;
%    end
% end
% 
% display(maxresult)
% display(block)
% 
% display(maxi)
% display(corresponding_block)
% display(curr_plane)
% 
% %******************************END******************************
% 
% %************EMBEDDING THE CONJUGATION MAP********************
% 
% % Embedding the conjugation map IN corresponding_block i.e.
% % the block which have maximum entropy in curr_plane i.e. bit plane number.
% n=1;
% for i=1:16
%     j=n;
%     
%     m1=floor((corresponding_block-1)/32)*8;
%     m2=m1+7;
%     k3=mod((corresponding_block-1),32);
%     k1=floor(k3)*8+1;
%     k2=k1+7;
%     for m=m1:m2
%         for p=k1:k2
%            curr_plane(m,p)=conmap(j);
%            j=j+1;
%         end
%     end
%     corresponding_block=corresponding_block+1;
%     n=j-1;
% end
%*******************END***********************

M=zeros(size(A));
M=bitset(M,8,bitget(A,8));
M=bitset(M,7,bitget(A,7));
M=bitset(M,6,bitget(A,6));
M=bitset(M,5,bitget(A,5));
M=bitset(M,4,bitget(A,4));
M=bitset(M,3,bitget(A,3));
M=bitset(M,2,bitget(A,2));
M=bitset(M,1,bitget(A,1));
M=uint8(M);
figure,imshow(M);

xy=zeros(256,256);
for i=1:256
xy(i,1)=M(i,1); 
end 
for i=1:256
for j=2:256
xy(i,j)=bitxor(M(i,j),xy(i,j-1)); 
end 
end 
pbc=uint8(xy);
title('CGC Image after embedding');
figure,subplot(1,2,1);imshow(pbc);
title('Original Image');
subplot(1,2,2);imshow(x);
title('Image after embedding secret file');