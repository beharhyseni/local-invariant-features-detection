from PIL import Image, ImageDraw
import numpy as np
import csv
import math

def ReadKeys(image):
    """Input an image and its associated SIFT keypoints.

    The argument image is the image file name (without an extension).
    The image is read from the PGM format file image.pgm and the
    keypoints are read from the file image.key.

    ReadKeys returns the following 3 arguments:

    image: the image (in PIL 'RGB' format)

    keypoints: K-by-4 array, in which each row has the 4 values specifying
    a keypoint (row, column, scale, orientation).  The orientation
    is in the range [-PI, PI] radians.

    descriptors: a K-by-128 array, where each row gives a descriptor
    for one of the K keypoints.  The descriptor is a 1D array of 128
    values with unit length.
    """
    im = Image.open(image+'.pgm').convert('RGB')
    keypoints = []
    descriptors = []
    first = True
    with open(image+'.key','rb') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC,skipinitialspace = True)
        descriptor = []
        for row in reader:
            if len(row) == 2:
                assert first, "Invalid keypoint file header."
                assert row[1] == 128, "Invalid keypoint descriptor length in header (should be 128)."
                count = row[0]
                first = False
            if len(row) == 4:
                keypoints.append(np.array(row))
            if len(row) == 20:
                descriptor += row
            if len(row) == 8:
                descriptor += row
                assert len(descriptor) == 128, "Keypoint descriptor length invalid (should be 128)."
                #normalize the key to unit length
                descriptor = np.array(descriptor)
                descriptor = descriptor / math.sqrt(np.sum(np.power(descriptor,2)))
                descriptors.append(descriptor)
                descriptor = []
    assert len(keypoints) == count, "Incorrect total number of keypoints read."
    print "Number of keypoints read:", int(count)
    return [im,keypoints,descriptors]

def AppendImages(im1, im2):
    """Create a new image that appends two images side-by-side.

    The arguments, im1 and im2, are PIL images of type RGB
    """
    im1cols, im1rows = im1.size
    im2cols, im2rows = im2.size
    im3 = Image.new('RGB', (im1cols+im2cols, max(im1rows,im2rows)))
    im3.paste(im1,(0,0))
    im3.paste(im2,(im1cols,0))
    return im3

def DisplayMatches(im1, im2, matched_pairs):
    """Display matches on a new image with the two input images placed side by side.

    Arguments:
     im1           1st image (in PIL 'RGB' format)
     im2           2nd image (in PIL 'RGB' format)
     matched_pairs list of matching keypoints, im1 to im2

    Displays and returns a newly created image (in PIL 'RGB' format)
    """
    im3 = AppendImages(im1,im2)
    offset = im1.size[0]
    draw = ImageDraw.Draw(im3)
    for match in matched_pairs:
        draw.line((match[0][1], match[0][0], offset+match[1][1], match[1][0]),fill="red",width=2)
    im3.show()
    return im3

def match(image1,image2):
    """Input two images and their associated SIFT keypoints.
    Display lines connecting the first 5 keypoints from each image.
    Note: These 5 are not correct matches, just randomly chosen points.

    The arguments image1 and image2 are file names without file extensions.

    Returns the number of matches displayed.

    Example: match('scene','book')
    """
    im1, keypoints1, descriptors1 = ReadKeys(image1)
    im2, keypoints2, descriptors2 = ReadKeys(image2)
    
   
   
   
   
   # PART 3 - SIFT Match 
    matched_pairs = []
    
    threshold = 0.65
    
    for desc1_index in range(len(descriptors1)):
        
        desc1_vector = descriptors1[desc1_index]
        angles_list = []
        
        for desc2_index in range(len(descriptors2)):
           desc2_vector = descriptors2[desc2_index]
           dot_product = np.dot(desc1_vector, desc2_vector)
           angle = math.acos(dot_product)
           angles_list.append(angle)
        
        
        best_angle = sorted(angles_list)[0]
        second_best_angle = sorted(angles_list)[1]
        
        ratio =  best_angle / second_best_angle
        
        if ratio < threshold:
            matched_pairs.append([keypoints1[desc1_index], keypoints2[angles_list.index(best_angle)]])
  
  
  
    # PART 4 - RANSAC
    
    RANSAC_ITERATIONS = 10
    
    orientation_degree = 30
    scale_agreement = 0.5
    
    
    consistent_subsets = []
    
    
    for i in range(0, RANSAC_ITERATIONS-1):
        
        random_index = np.random.randint(0, len(matched_pairs) - 1, 1)
        
        random_matched_pair = matched_pairs[random_index[0]]
        
        
        first_pair_degree = math.degrees(abs(random_matched_pair[0][3] - random_matched_pair[1][3]))
        first_pair_scale = abs(random_matched_pair[0][2] - random_matched_pair[1][2])
       
        
        consistent_subset = []
        # print len(consistent_subset)

        consistent_subset.append(random_matched_pair)
        
        for pair_index in range(0, len(matched_pairs) - 1):
            
            
            second_pair_degree = math.degrees(abs(matched_pairs[pair_index][0][3] - matched_pairs[pair_index][1][3]))
            second_pair_scale  = abs(matched_pairs[pair_index][0][2] - matched_pairs[pair_index][1][2])
            
            change_in_orientation = abs(first_pair_degree - second_pair_degree)
            change_in_scale       = abs(first_pair_scale - second_pair_scale)
            
           
            change_in_scale_agreement = (change_in_scale <= (first_pair_scale + scale_agreement * first_pair_scale)) and (change_in_scale >= (first_pair_scale - scale_agreement * first_pair_scale))
           
            
            if (change_in_orientation < orientation_degree) and change_in_scale_agreement:
                 #print matched_pairs[pair_index]
                 consistent_subset.append(matched_pairs[pair_index])
                 
        
        consistent_subsets.append(consistent_subset)
    
        
    # now check which subsets have the most elements, which means the pairs were the most consistent
        
    
    consistent_matched_pairs = []
    
    consistent_pairs_index = 0
    subset_length = 0
    for i in range(0, len(consistent_subsets)-1):
        
        if subset_length < len(consistent_subsets[i]):
            subset_length = len(consistent_subsets[i]) - 1
            consistent_pairs_index = i
            
        
        
    for i in range(0, len(consistent_subsets[consistent_pairs_index]) - 1):
        consistent_matched_pairs.append(consistent_subsets[consistent_pairs_index][i])
        
    
    
    # END OF SECTION OF CODE TO REPLACE
    #
    im3 = DisplayMatches(im1, im2, consistent_matched_pairs)
    return im3

#Test run...
# match('scene','basmati')
# match('scene', 'book')
# match('scene', 'box')
match('library2','library')

