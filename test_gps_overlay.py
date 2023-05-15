import cv2
import numpy as np
import matplotlib.path as mpltPath

DIR = 'D:/Data/TruckSimulator/roof_cam/raw/Peterbilt/sess_07/clip_7/drive.mp4'

video = cv2.VideoCapture(DIR)

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

def sliding_window(img, nwindows=6, margin=100, minpix = 1, draw_windows=True):
    a, b, c = [],[],[]
    fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    histogram = get_hist(img)
    # Find peaks of halves
    midpoint = int(histogram.shape[0]/2)
    x_base = np.argmax(histogram[:midpoint])
    
    
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    x_current = x_base
    
    
    # Create empty lists to receive lane pixel indices
    lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv2.rectangle(out_img,(win_x_low,win_y_low),(win_x_high,win_y_high),
            (100,255,255), 3)
        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]
        # Append these indices to the lists
        lane_inds.append(good_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))

    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)

    # Extract left and right line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]

    # Fit a second order polynomial to each
    fit = np.polyfit(y, x, 2)
    
    a.append(fit[0])
    b.append(fit[1])
    c.append(fit[2])
    
    fit_[0] = np.mean(a[-10:])
    fit_[1] = np.mean(b[-10:])
    fit_[2] = np.mean(c[-10:])
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    fitx = fit_[0]*ploty**2 + fit_[1]*ploty + fit_[2]

    out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [255, 0, 100]
    
    return out_img, fitx, fit_, ploty

def draw_lane(img, fit, ploty):
    color_img = np.zeros_like(img)
    
    line = np.array(np.transpose(np.vstack([fit, ploty])), dtype=np.int)

    cv2.polylines(color_img, [line], False, (0,255,0), 10)

    return color_img

def perspective_warp(img, roi, dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    src = np.float32(roi)
    dst_size = (img.shape[1], img.shape[0])
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def inv_perspective_warp(img, roi, src=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    dst = np.float32(roi)
    dst_size = (img.shape[1], img.shape[0])
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

region_of_interest = [(0.15,0),(0.85,0),(0,1),(1,1)]

while 1:
    ret, frame = video.read()

    if not ret:
        break

    frame_region = cv2.rectangle(frame.copy(), (310,380), (970,700), (0,255,0), 2)
    region = frame[380:700, :, :]

    #h, w, _ = region.shape
    #draw_region_of_interest = [region_of_interest[0], region_of_interest[1], region_of_interest[3], region_of_interest[2]]
    #cv2.polylines(region, [np.array(np.float32(draw_region_of_interest)*np.float32((w,h)), np.int32)], True, (255,0,0), 1)
    #frame_region[400:700, :, :] = region

    gps = frame[538:565, 1050:1200, :]

    h, w, _ = gps.shape
    gps = cv2.resize(gps, (w*4, h*4))
    h, w, _ = gps.shape

    gps_blue = gps[:,:,0]
    gps_green = gps[:,:,1]
    gps_red = gps[:,:,2]

    gps_green_b = np.zeros_like(gps_green)
    gps_red_b = np.zeros_like(gps_red)
    gps_green_b[(gps_blue < 50) & (gps_green > 130) & (gps_red < 50)] = 1
    gps_red_b[(gps_blue < 50) & (gps_green < 50) & (gps_red > 130)] = 1

    gps_mask = np.zeros_like(gps_red)
    gps_mask[(gps_red_b == 1)] = 255

    mask = np.zeros((h+2, w+2), np.uint8)
    holes = gps_mask.copy()
    cv2.floodFill(holes, mask, (0,0), 255)
    holes = cv2.bitwise_not(holes)
    gps_mask = cv2.bitwise_or(gps_mask, holes)

    sliding, curve, _, ploty = sliding_window(gps_mask, margin=70, nwindows=15, draw_windows=True)

    path = draw_lane(gps, curve, ploty)

    path_overlay = inv_perspective_warp(gps, roi=region_of_interest)
    rh, rw, _ = region.shape

    path_on_frame = cv2.bitwise_or(region, cv2.resize(path_overlay, (rw,rh)))

    cv2.imshow('Frame', frame)
    cv2.imshow('Region', frame_region)
    #cv2.imshow('Birdseye', perspective_warp(region, roi=region_of_interest))
    #cv2.imshow('GPS', gps)
    #cv2.imshow('GPS Mask', gps_mask)
    #cv2.imshow('Holes', holes)
    #cv2.imshow('Sliding', sliding)
    #cv2.imshow('Path', path) 
    cv2.imshow('Overlay', path_on_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('test_frame.jpg', frame)
        break
    
#cv2.waitKey(0)