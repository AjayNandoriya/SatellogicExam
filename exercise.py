import numpy as np
import cv2
from matplotlib import pyplot as plt
import timeit
from skimage import color, data, restoration


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        if len(image.shape) == 2:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        row,col,ch= image.shape
        mean = 0
        var = 0.01
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        # vals = len(np.unique(image))
        # vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(0.001, image.shape) + image
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

def cart2sph(point3d):
    x = point3d[0]
    y = point3d[1]
    z = point3d[2]
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def GetRotationMatrix(az, el, th = 0):
    e = np.zeros((3,1))
    e[0], e[1], e[2] = sph2cart(az, el, 1)
    e = -e
    ecross = np.outer(e, e)
    I3 = np.eye(3)
    R = I3*np.cos(th) + (1 - np.cos(th))*np.inner(e,e) + ecross*np.sin(th)
    return R

def GetRotationMatrix(az=0, el=0):
    satellite_Rotation_matrix = np.eye(3)
    cos_az = np.cos(az)
    sin_az = np.sin(az)
    cos_el = np.cos(el)
    sin_el = np.sin(el)
    satellite_Rotation_matrix[0, :] = np.array([-sin_az, cos_az, 0])
    satellite_Rotation_matrix[1, :] = np.array([-cos_az*sin_el, -sin_az*sin_el, cos_el])
    satellite_Rotation_matrix[2, :] = np.array([-cos_az*cos_el, -sin_az*cos_el, -sin_el])
    return satellite_Rotation_matrix
    
def GetSatellitePositionFromGround(ground_point, d=450*1000):
    az, el, R = cart2sph(ground_point)
    satellite_direction_z = -ground_point/R
    # satellite_position = -(R+d)*satellite_direction_z
    satellite_position = GetSatellitePosition(R=R, d=d, az=az, el=el)
    return satellite_position, satellite_direction_z

def GetSatellitePosition(R = 6400*1000, d = 450*1000, az = 0, el = 0):
    satellite_focal_point = np.reshape(np.array(sph2cart(az, el, R + d)),(3,1))
    return satellite_focal_point

def GetPixelVector(x, y, f = 2, pixelsize = 5e-6, az = 0, el = 0):
    satellite_rot_mat = GetRotationMatrix(az, el)
    pixel3D = np.reshape(np.array([x*pixelsize, y*pixelsize, -f]),(3,1))
    pixel_vector = satellite_rot_mat.T.dot(pixel3D)
    return pixel_vector

def Convert_2D_to_3D(x, y, R = 6400*1000, d = 450*1000, f = 2, pixelsize = 5e-6, az = 0, el = 0):
    ground_point = np.array(sph2cart(az, el, R))
    satellite_focal_point, dir_z = GetSatellitePositionFromGround(ground_point)
    pixel_vector = GetPixelVector(x=x, y=y, f=f, pixelsize=pixelsize, az=az, el=el)
    point3D = satellite_focal_point + pixel_vector
    return point3D

def Convert_2D_to_surface(x,y,R = 6400*1000, d = 450*1000, f = 2, pixelsize = 5e-6, az = 0, el = 0):
    ground_point = np.array(sph2cart(az, el, R))
    satellite_focal_point, dir_z = GetSatellitePositionFromGround(ground_point)
    satellite_rot_mat = GetRotationMatrix(az, el)
    Np = len(x)
    surface_point3D = np.zeros((3,Np))
    for i in range(Np):
        pixel3Ds = np.reshape(np.array([x[i]*pixelsize, y[i]*pixelsize, -f]),(3,1))
        pixel_vector = satellite_rot_mat.T.dot(pixel3Ds)
        # pixel_vector = GetPixelVector(x=x, y=y, f=f, pixelsize=pixelsize, az=az, el=el)
        point3D = satellite_focal_point + pixel_vector
        V2 = pixel_vector.T.dot(pixel_vector)
        VF = pixel_vector.T.dot(satellite_focal_point)
        F2 = satellite_focal_point.T.dot(satellite_focal_point)
        R2 = R*R
        k = (-2*VF + np.sqrt(4*VF*VF -4*V2*(F2-R2)))/(2*V2)
        surface_point3D[:,i] = np.reshape(satellite_focal_point + k*(pixel_vector),(3))
    return surface_point3D

def GetFocalLength(d=450*1000, pixelsize = 5e-6, resolution=1):
    focal_length = d*pixelsize/resolution
    # delta_f === delta_d*pixelsize/resolution
    return focal_length

# Question-2:
# for given site, estimate PSF using L2-norm cost function optimization
# after estimation PSf, use iterative method such as Richardson?Lucy deconvolution for correction
# also, if there is intensity error due to atmosphere then registration using edges can be performed first

def EstimatePSF(img, img_out):
    # here I am esitmating psf using frequency domain but better solution would be to estimate using iterative solver with L2-norm or L1-norm optimization assuming we have
    # high quality image as reference. if such image is not available then due to known pattern, we can generate synthetic image as reference or use edge sharpness for optimization
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    f_out = np.fft.fft2(img_out)
    fshift_out = np.fft.fftshift(f_out)
    fshift_psf = np.abs(fshift_out)/(np.abs(fshift) + 0.001)
    f_ishift = np.fft.ifftshift(fshift_psf)
    psf_kernel = np.fft.ifftshift(np.abs(np.fft.ifft2(f_ishift)))

    k_shape = psf_kernel.shape
    N = 5
    psf_kernel = psf_kernel[int(k_shape[0]/2 -N+1):int(k_shape[0]/2 + N + 1),int(k_shape[1]/2 -N + 0):int(k_shape[1]/2 +N + 0)]
    
    psf_norm = np.sum(psf_kernel)
    psf_kernel /= psf_norm
    return psf_kernel


def norm(img):
    s2 = np.multiply(img,img)
    if len(img.shape) == 3:
        n = np.sqrt(np.mean(s2[10:-10,10:-10,:]))
    else:
        n = np.sqrt(np.mean(s2))
    return n


def test_convert2dto3d():
    start = timeit.default_timer()
    Nx = 5000
    Ny = 100
    img = np.zeros((Ny*2+1, Nx*2+1))
    x = np.linspace(-Nx,Nx,2*Nx+1)
    y = np.linspace(-Ny,Ny,2*Ny+1)
    xv,yv = np.meshgrid(x,y)
    ix = xv.flatten()
    iy = yv.flatten()
    point3Ds = Convert_2D_to_surface(x = ix, y = iy)
    psf_img = cv2.imread("sample3.png", 0)
    point3Ds[1,:] += (psf_img.shape[1])/2
    point3Ds[2,:] += (psf_img.shape[0])/2 
    for i in range(point3Ds.shape[1]):
        if point3Ds[1,i]< 0 or point3Ds[1,i] > psf_img.shape[1]:
            continue
        if point3Ds[2,i]< 0 or point3Ds[2,i] > psf_img.shape[0]:
            continue
        val = psf_img[int(point3Ds[2,i]), int(point3Ds[1,i])] 
        img[int(iy[i]+Ny), int(ix[i]+Nx)] = val
            # az, el, r = cart2sph(point3D)
            # print("({0},{1}) --> ({2},{3},{4}) ({5},{6},{7})\n".format(ix, iy, point3D[0], point3D[1], point3D[2], az, el, r))
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    cv2.imwrite("sample3_satellite.png",img)  
    plt.imshow(img)
    plt.show()

def question1():
    print("Question 1:")
    # I am not familiar with equation for noise to exposure time relation
    # as exposure time increases, noise decreases but not sure it is 1/t polynomial or exp(-t)
    # I will need to check theory first 
    # once we have that relation, we can form overall noise formula for weighted sum and optimize square mean cost function
    # but as per my current understanding, if HE has low noise than
    img = cv2.imread("calibration2.png",0)
    img = img.astype(np.float32)/255
    I_le = noisy("gauss", img)[:,:,0]
    I_he = noisy("gauss", img*1.1)[:,:,0]
    I_he[I_he<0] = 0
    I_he[I_he>1] = 1
    mask = I_he< 1
    inv_mask = np.logical_not(mask)
    scale = np.mean(I_he[mask])/np.mean(I_le[mask])
    I_he_corrected = I_he/ scale
    I_he_corrected[inv_mask] = I_le[inv_mask]
    
    cov_mat = np.cov(np.vstack((I_he_corrected.flatten(),I_le.flatten())))
    std_he_2 = cov_mat[0,0]
    std_le_2 = cov_mat[1,1]
    std_le_he = cov_mat[0,1]
    
    # weighted sum of 2 gaussian variable
    # assuming mean of both signal is same then optimizing for variance
    w = (std_he_2 - std_le_he)/(std_le_2 + std_he_2 -2*std_le_he)
    I_out = w*I_le + (1-w)*I_he_corrected

    le_err = norm(I_le - img)
    he_err = norm(I_he - img)
    he_corrected_err = norm(I_he_corrected - img)
    out_err = norm(I_out - img)
    print("w = {0}, scale = {1}, err: le={2} , he={3},he_corrected={4} out = {5}".format(w,scale,le_err,he_err,he_corrected_err,out_err))
    ax1 = plt.subplot(321)
    plt.imshow(img), plt.title("image")
    plt.subplot(322, sharex=ax1, sharey=ax1), plt.imshow(I_le),plt.title("Low Exposure : err ={0}".format(le_err))
    plt.subplot(323, sharex=ax1, sharey=ax1), plt.imshow(I_he),plt.title("high Exposure : err ={0}".format(he_err))
    plt.subplot(324, sharex=ax1, sharey=ax1), plt.imshow(inv_mask),plt.title("high Exposure saturation mask")
    plt.subplot(325, sharex=ax1, sharey=ax1), plt.imshow(I_he_corrected),plt.title("high Exposure corrected : err ={0}".format(he_corrected_err))
    plt.subplot(326, sharex=ax1, sharey=ax1), plt.imshow(I_out),plt.title("Out Image : w ={0}, err = {1}".format(w,out_err))
    plt.show()

def question2():
    print("Question 2:")
    psf = np.load("example_psf.npy")
    img = cv2.imread("calibration2.png")
    img = img.astype(np.float32)/255
    psf_img = cv2.filter2D(img,-1,psf)
    psf_img_noise_val = norm(psf_img - img)
    
    noise_img1 = noisy("gauss", psf_img)
    noise_img = noisy("poisson", noise_img1)
    noise2_val = norm(noise_img - img)
    
    psf_estimated =EstimatePSF(img[:,:,0], noise_img[:,:,0])
    filtered_img = np.copy(noise_img)
    for k in range(img.shape[2]):
        # filtered_img[:,:,k] = cv2.filter2D(noise_img[:,:,k],-1,psf_estimated)
        filtered_img[:,:,k], _ = restoration.unsupervised_wiener(noise_img[:,:,k], psf_estimated)
    filtered_noise_val = norm(filtered_img - img)
    
    ax1 = plt.subplot(2,2,1)
    plt.imshow(img)
    plt.subplot(2,2,2, sharex=ax1, sharey=ax1),plt.imshow(psf_estimated)
    plt.subplot(2,2,3, sharex=ax1, sharey=ax1),plt.imshow(noise_img),plt.title(noise2_val)
    plt.subplot(2,2,4, sharex=ax1, sharey=ax1),plt.imshow(filtered_img), plt.title(filtered_noise_val)
    
    plt.show()


def question3():
    print("Question 3:")
    psf = np.load("example_psf.npy")
    print(psf)
    img = cv2.imread("calibration2.png")
    # scale_img = cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4)))
    # cv2.imwrite("calibration2.png", scale_img)
    img = img.astype(np.float32)/255
    psf_img = cv2.filter2D(img,-1,psf)
    psf_img_noise_val = norm(psf_img - img)
    # noise_img1 = psf_img
    # noise_img = noise_img1
    noise_img1 = noisy("gauss", psf_img)
    noise_img = noisy("poisson", noise_img1)
    
    noise1_val = norm(noise_img1 - img)
    noise2_val = norm(noise_img - img)
    
    filtered_img = np.copy(noise_img)
    for k in range(filtered_img.shape[2]):
        # filtered_img[:,:,k] = restoration.richardson_lucy(noise_img[:,:,k], psf, iterations=30)
        filtered_img[:,:,k], _ = restoration.unsupervised_wiener(noise_img[:,:,k], psf)

    filtered_noise_val = norm(filtered_img - img)
    cv2.imwrite("calibration_after_psf.png", psf_img*255)
    cv2.imwrite("calibration_after_noise.png", noise_img*255)
    cv2.imwrite("calibration_after_filter.png", filtered_img*255)
    ax1 = plt.subplot(2,2,1)
    plt.imshow(img), plt.title("Input Image")
    plt.subplot(2,2,2, sharex=ax1, sharey=ax1),plt.imshow(psf_img), plt.title("img after psf: {0}".format(psf_img_noise_val))
    plt.subplot(2,2,3, sharex=ax1, sharey=ax1),plt.imshow(noise_img),plt.title("img with noise : {0}".format(noise2_val))
    plt.subplot(2,2,4, sharex=ax1, sharey=ax1),plt.imshow(filtered_img), plt.title("filtered image : {0}".format(filtered_noise_val))
    plt.show()

def question4():
    print("Question 4:")
    start = timeit.default_timer()
    Nx = 5000
    Ny = 100
    img = np.zeros((Ny*2+1, Nx*2+1))
    x = np.linspace(-Nx,Nx,2*Nx+1)
    y = np.linspace(-Ny,Ny,2*Ny+1)
    xv,yv = np.meshgrid(x,y)
    ix = xv.flatten()
    iy = yv.flatten()
    point3Ds = Convert_2D_to_surface(x = ix, y = iy)
    psf_img = cv2.imread("sample3.png", 0)
    point3Ds[1,:] += (psf_img.shape[1])/2
    point3Ds[2,:] += (psf_img.shape[0])/2 
    for i in range(point3Ds.shape[1]):
        if point3Ds[1,i]< 0 or point3Ds[1,i] > psf_img.shape[1]:
            continue
        if point3Ds[2,i]< 0 or point3Ds[2,i] > psf_img.shape[0]:
            continue
        val = psf_img[int(point3Ds[2,i]), int(point3Ds[1,i])] 
        img[int(iy[i]+Ny), int(ix[i]+Nx)] = val
            # az, el, r = cart2sph(point3D)
            # print("({0},{1}) --> ({2},{3},{4}) ({5},{6},{7})\n".format(ix, iy, point3D[0], point3D[1], point3D[2], az, el, r))
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    cv2.imwrite("sample3_satellite.png",img)  
    

def question5():
    print("Question 5:")
    d=450*1000
    pixelsize = 5e-6
    resolution=1
    # focal_length = d*pixelsize/resolution
    # assuming satellite line of sight exact vertical & discarding earth curvature for simplification
    focal_length = GetFocalLength(d=d,pixelsize=pixelsize, resolution=resolution)
    print("focal length @450km: {0}\n".format(focal_length) )

    d -= 100
    focal_length_neg = GetFocalLength(d=d,pixelsize=pixelsize, resolution=resolution)
    d += 200
    focal_length_pos = GetFocalLength(d=d,pixelsize=pixelsize, resolution=resolution)
    print("focal length @450km - 100m: {0}\n".format(focal_length_neg) )
    print("focal length @450km + 100m: {0}\n".format(focal_length_pos) )

    # part-3
    # compensate resolution variation using by estimating scale factor across all capture images
    # cost =sum(L2(pixelsize*d_i -focal_length*resolution_i))
    # --> focal_length = pixelsize*mean(d) / mean(resolution)

    # part-4
    # It is not clear to me how to use question-4 2D-3D projection data for focal_length estimation



def test_settelite_position():
    ground_point = np.array(sph2cart(np.pi/4,0, 6400*1000))
    satellite_position, satellite_dir_z, satellite_rot_mat = GetSatellitePosition(ground_point)
    print("ground point", ground_point)
    print("pos", satellite_position)
    print("dir_z", satellite_dir_z)
    print("rot mat", satellite_rot_mat)

def createImg(Nx,Ny):
    img = np.zeros((Ny,Nx))
    cv2.imwrite("sample.png",img)

def blurImg():
    psf_img = cv2.imread("sample2.png")
    psf_fimg = cv2.blur(psf_img,(100,3))
    cv2.imwrite("sample3.png",psf_fimg)
    
if __name__ == "__main__":
    # blurImg()
    # createImg(10001,201)
    question1()
    question2()
    question3()
    question4()
    question5()