import cv2
import numpy as np
from PIL import Image
import pytesseract
import os
pytesseract.pytesseract.tesseract_cmd=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

global detected_text,col
def delete_folder_contents(folder_path):
    # Iterate over all files and directories in the folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            # Delete the file
            os.remove(item_path)
        elif os.path.isdir(item_path):
            # Delete the directory and its contents recursively
            delete_folder_contents(item_path)
            os.rmdir(item_path)

def extract_text_blocks(block1, block2, x):
    img1 = block1.copy()

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    hist = cv2.reduce(gray, 1, cv2.REDUCE_AVG).reshape(-1)

    peaks = np.where(hist < 10)[0]

    min_block_height = 2

    block_positions = []

    detected_texts = ""  # Initialize an empty string to store detected texts

    for i in range(len(peaks) - 1):
        y1 = peaks[i]
        y2 = peaks[i+1]
        block_height = y2 - y1
        if block_height > min_block_height:
            block_positions.append((y1, y2))

    img2 = block2.copy()

    if len(block_positions) == 0:
        detected_text = pytesseract.image_to_string(img2)
        #cv2.imshow(f"Segmented_Vign_{x}.jpg", img2)
        cv2.imwrite(f"Detected Lines/Segmented_Vign_{x}.jpg", img2)
        if detected_text is not None:
            detected_texts += detected_text + "\n"  # Append detected text to the string
    else:
        for i, (y1, y2) in enumerate(block_positions):
            y1_new = max(y1 - 10, 0)
            y2_new = min(y2 + 10, img2.shape[0])
            block = img2[y1_new:y2_new, 0:img1.shape[1]]
            detected_text = pytesseract.image_to_string(block)
            #cv2.imshow(f"Segmented_Vign_{x}_{i+1}.jpg", block)
            cv2.imwrite(f"Detected Lines/Segmented_Vign_{x}_{i+1}.jpg", block)

            if detected_text is not None:
                detected_texts += detected_text + "\n"  # Append detected text to the string

    return detected_texts  # Return the string of detected texts

def process_images(img1_path, img2_path):
    delete_folder_contents("Detected Lines")
    delete_folder_contents("Detected Parts")
    # Load the first image and convert to grayscale
    img1 = cv2.imread(img1_path)
    img1 = cv2.resize(img1, (800, 300))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Compute vertical histogram
    vert_hist = cv2.reduce(gray1, 0, cv2.REDUCE_AVG).reshape(-1)

    # Set threshold for vertical text fields
    vert_thresh = 1

    # Draw a line when there are 5 or more columns with average pixel value below the threshold
    line_started = False
    line_start = 0
    line_positions = []
    for i in range(len(vert_hist)):
        if vert_hist[i] < vert_thresh:
            if not line_started:
                line_started = True
                line_start = i
            elif i - line_start >= 5:
                line_positions.append(line_start)
                line_started = False
        else:
            line_started = False

    # Add the last line position
    line_positions.append(img1.shape[1])

    # Save the block positions from the first image
    block_positions1 = []
    min_block_height = 20
    for i in range(len(line_positions)-1):
        y1 = line_positions[i]
        y2 = line_positions[i+1]
        block_height = y2 - y1
        if block_height > min_block_height:
            block_positions1.append((y1, y2))

    # Load the second image and convert to grayscale
    img2 = cv2.imread(img2_path)
    img2 = cv2.resize(img2, (800, 300))
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    detected_text = ""  # Initialize an empty string to store detected texts

    # Apply the block positions to the first image
    for i, block_pos in enumerate(block_positions1):
        y1, y2 = block_pos
        block1 = img1[0:img1.shape[0], y1:y2]
        block2 = img2[0:img2.shape[0], y1:y2]
        block_shape1 = block1.shape
        block_shape2 = block2.shape
        if block_shape1[0] > block_shape1[1]:
            block1 = cv2.rotate(block1, cv2.ROTATE_90_CLOCKWISE)
        if block_shape2[0] > block_shape2[1]:
            block2 = cv2.rotate(block2, cv2.ROTATE_90_CLOCKWISE)

        #cv2.imshow(f"Black_White_{i+1}.jpg", block1)
        cv2.imwrite(f"Detected Parts/Black_White_{i+1}.jpg", block1)
        #cv2.imshow(f"Part_{i+1}.jpg", block2)
        cv2.imwrite(f"Detected Parts/Part_{i+1}.jpg", block2)

        detected_texts = extract_text_blocks(block1, block2, i)
        detected_text += detected_texts + "\n"  # Append detected texts to the string

    #print(detected_text)
    return detected_text  # Return the string of detected texts

def areaFilter(minArea, inputImage):
    # Perform an area filter on the binary blobs:
    componentsNumber, labeledImage, componentStats, componentCentroids = \
        cv2.connectedComponentsWithStats(inputImage, connectivity=4)
    
    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]

    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    return filteredImage

def detection(event, x, y, flags, param):
    global src, ori_img, col, img, result
    
    if event == cv2.EVENT_LBUTTONUP:
        img = ori_img.copy()
        src.append([x, y])
        for xx, yy in src:
            cv2.circle(img, center=(xx, yy), radius=5, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
        cv2.imshow('img', img)
        if len(src) == 4:       
            src_np = np.array(src, dtype=np.float32)       
            width = max(np.linalg.norm(src_np[0] - src_np[1]), np.linalg.norm(src_np[2] - src_np[3]))
            height = max(np.linalg.norm(src_np[0] - src_np[3]), np.linalg.norm(src_np[1] - src_np[2]))

           
            dst_np = np.array([
                [0, 0],
                [width, 0],
                [width, height],
                [0, height]
            ], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src=src_np, dst=dst_np)
            result = cv2.warpPerspective(ori_img, M=M, dsize=(int(width), int(height)))
            result = cv2.resize(result, (600, 400))    
            img = np.copy(result)
            #cv2.imshow('img', img)
            cv2.imwrite('ResVign.jpg', result)
            #print(f"Detected color: {col}")
            inputImage=cv2.imread("ResVign.jpg")
            imgFloat = inputImage.astype(float) / 255.

            kChannel = 1 - np.max(imgFloat, axis=2)

            kChannel = (255 * kChannel).astype(np.uint8)

            binaryThresh = 150
            _, binaryImage = cv2.threshold(kChannel, binaryThresh, 255, cv2.THRESH_BINARY)
            minArea = 100
            binaryImage = areaFilter(minArea, binaryImage)

            kernelSize = 3


            opIterations = 2

            morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))


            binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)



            #cv2.imshow("binaryImage [closed]", binaryImage)
            cv2.imwrite("Black_Image.jpg",binaryImage)
            detected_text=process_images('Black_Image.jpg', 'ResVign.jpg')
            with open('dt2.txt', 'w') as file:
              file.write(detected_text)

def is_red_or_green(filename):
    img = Image.open(filename)
    
    
    pixels = img.load()
    width, height = img.size
    
    
    red_range = (150, 255, 0, 120, 0, 120)
    green_range = (0, 120, 150, 255, 0, 120)
    
   
    red_count = 0
    green_count = 0
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x,y]
            if (r > red_range[0] and r < red_range[1] and
                g > red_range[2] and g < red_range[3] and
                b > red_range[4] and b < red_range[5]):
                red_count += 1
            elif (r > green_range[0] and r < green_range[1] and
                  g > green_range[2] and g < green_range[3] and
                  b > green_range[4] and b < green_range[5]):
                green_count += 1
                

    if red_count > green_count:
        return "Red(Non-Refundable)"
    elif green_count > red_count:
        return "Green(Refundable)"
    else:
        return "White(Non-Refundable)"
    

def Tesseract_based(uploaded_image_path):
    global ori_img,detected_text, src 
    ori_img = cv2.imread(uploaded_image_path)
    ori_img = cv2.resize(ori_img, (600, 600))
    src = []
    res = []
    cv2.namedWindow('img')
    
    cv2.setMouseCallback('img', detection)
    cv2.imshow('img', ori_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()