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
    
    
# HELPER FUNCTION
def correct_degrees(degrees_difference):
      value = degrees_difference
      
      # if the degrees_difference is larger than pi (180 degrees), than it we modify it to be 360 - (degrees_difference), since we need the degrees to be at most 180 degrees.
      if abs(degrees_difference) > 180:
            value = 360 - degrees_difference
      return value
    

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
    
   
    # *** PART 3 - SIFT Match *** 
   
    # Initialize an array which will save matched pairs found using SIFT to detect same features between 2 pictures.
    matched_pairs = []
    
    # Set the threshold value, which is used to filter matches by comparing their angles (the smallest (best) match's angle to the second-best match's angle); thus, eliminate false matches.
    threshold = 0.75

    # Iterate through every element of the descriptors1 array that corresponds to the first image's descriptor vectors 
    for desc1_index in range(0, len(descriptors1)):
        
        # Save the current (based on the current iteration of descriptors1) descriptor vector of image1 into the variable desc1_vector
        # It will be used to compute the dot product with vectors from descriptor2 (from image 2)
        desc1_vector = descriptors1[desc1_index]
        
        # Initialize a list variable that will save angles (to be used to save the inverese cosine of the dot product 
        # between vectors of image1 and image2 descriptors (Each row of descriptors corresponds to a descriptor vector)
        angles_list = []
        
        # Iterate through every element of the descriptors2 array that corresponds to the second image's descriptor vectors 
        for desc2_index in range(0, len(descriptors2)):
            
           # Save the current (based on the current iteration of descriptors2) descriptor vector of image2 into the variable desc2_vector. 
           # It will be used to compute the dot product between desc2_vector and a vector from descriptor1 array (from image1) 
           desc2_vector = descriptors2[desc2_index]
                      
           # Compute the dot product between the current descriptor vector of the image1's descriptor1 and the current vector of the image2's descriptor2 array
           dot_product = np.dot(desc1_vector, desc2_vector)
           
           # Save the inverse cosine of the calculated dot product in the variable 'angle', which denotes the angle between the descriptor vectors from image1 and image2.
           angle = math.acos(dot_product)
           
           # Add this calculated angle (inverse cosine of the dot product between descriptor vectors of image1 and image2) to the list 'angles_list'
           angles_list.append(angle)
        
        # Sort the angles from the 'angles_list' in the increasing order. This is an easy way to locate and save 
        # the best calculated angle (smallest number) to the variable 'best_angle'.
        best_angle = sorted(angles_list)[0]
        
        # Locate and save the second best angle in the list, which is the second element in this sorted list (second smallest number of the list).
        second_best_angle = sorted(angles_list)[1]
        
        # We divide the best angle with the second best angle in the list to obtain the ratio between these two.
        ratio =  best_angle / second_best_angle
        
        # Compare the ratio with the specified threshold. If the ratio is smaller than the threshold, then add the correct keypoint of image1 (based on the current iteration descriptor vector index)
        # with the correct keypoint from the image1 as a matched pair into the matched pairs' array.
        # (The correct keypoint from image1 is obtained by using the current iteration's desc1_index as an index to get the corresponding element from the keypoints array of image1)
        # (The correct keypoint from image2 is obtained by getting the index of the best_angle from the angles_list (indexes in angles_list correspond to the indexes in keypoints of image2), 
        # then we use this index's value as an index for keypoints2 to obtain the correct keypoint).
        if ratio < threshold:
            matched_pairs.append([keypoints1[desc1_index], keypoints2[angles_list.index(best_angle)]])
  
  
  
    # *** PART 4 - RANSAC *** 
    
    # Specify the number of RANSAC iterations (10 is the number asked in the assignment description)
    RANSAC_ITERATIONS = 10
    
    # Orientation degree and scale variables (are manipulated differently to obtain best keypoint matching results using RANSAC)
    orientation_degree = 25
    scale_percentage = 0.55
    
    # Declare the list 'consistent_subsets' that will contain conistent matched pairs when comparing degrees and scales of the random matched pair with every other element in the 
    # matched_pair list.
    consistent_subsets = []
    
    
    # Iterate RANSAC_ITERATIONS times and for each iteration select just one match at random, and then check all the other matches for consistency with it.
    for i in range(RANSAC_ITERATIONS):
        
        # Select a random integer between 0 and the array length of the 'matched_pairs'. (which means a random index of an element from the matched pairs)
        random_index = np.random.randint(0, len(matched_pairs), 1)
        
        # Select the random matched pair from the matched_pairs list by using the random index we calculated before.
        random_matched_pair = matched_pairs[random_index[0]]
        
        # Calculate the difference of degrees between the two keypoints (2nd keypoint - first) in the random matched pair of this iteration (since each RANSAC_ITERATION, a different random match pair is chosen).
        # This degrees difference is computed by obtaining the orientations (the fourth element in a keypoint row) of the two keypoints in this matched pair, converting these radians in degrees and then finding
        # their difference by subtracting 1st keypoint degree from the 2nd keypoint degree.
        first_pair_degree = math.degrees(random_matched_pair[1][3]) - math.degrees(random_matched_pair[0][3])
     
        # Calculate the difference of scales between the two keypoints (2nd keypoint over 1st keypoint) in the random matched pair of this iteration (since each RANSAC_ITERATION, a different random match pair is chosen).
        # To compute this difference, we select the element in the third row (scale is positioned there) in each of the keypoints in this matched pair and find their absolute ratio by diving second keypoint's scale by first keypoint's scale.            
        first_pair_scale = abs(random_matched_pair[1][2] / random_matched_pair[0][2])
       
        # Declare the list 'consistent_subset' that will save consistent matches in each RANSAC iteration.
        consistent_subset = []

        # Add the first matched pair (the chosen random matched pair) in the consistent_subset list.
        consistent_subset.append(random_matched_pair)
    
        # Iterate through each element of the matched_pairs (pair_index is the index of the matched pair in the current iteration)
        # We do this iteration since we need to check all matched pairs in the matched_pairs list for consistency with the random chosen pair.
        for pair_index in range(0, len(matched_pairs)):
            
            # Calculate the difference of degrees between the two keypoints (2nd keypoint - first) in the matched pair of the current iteration of matched_pairs index. (since each iteration, a different pair_index is chosen)
            # This degrees difference is computed by obtaining the orientations (the fourth element in a keypoint row) of the two keypoints in this matched pair, converting these radians in degrees and then finding
            # their difference by subtracting 1st keypoint degree from the 2nd keypoint degree.
            second_pair_degree = math.degrees(matched_pairs[pair_index][1][3]) - math.degrees(matched_pairs[pair_index][0][3])
            
            # Calculate the difference of scales between the two keypoints (2nd keypoint over 1st keypoint) in the matched pair of the current iteration of matched_pairs index. (since each iteration, a different pair_index is chosen)
            # To compute this difference, we select the element in the third row (scale is positioned there) in each of the keypoints in this matched pair and find their absolute ratio by diving second keypoint's scale by first keypoint's scale.            
            second_pair_scale  = abs(matched_pairs[pair_index][1][2] / matched_pairs[pair_index][0][2])
            
            # Calculate the absolute difference between the degrees computed between the two keypoints in the random matched pair and the two keypoints in matched pair of the current pair_index iteration.
            # In other words, compute the absolute degrees difference between the random matched pair and the current iteration's matched pair that comes from matched_pairs list.
            change_in_orientation = correct_degrees(abs(second_pair_degree - first_pair_degree))
                       
            # Check the scale agreement of two the matched pairs (the random matched pair and the one from the current iteraiton of matched_pairs) and save the result as boolean which will be used in the later
            # if statement. In order for the change_in_scale_agreement to be true, the scale of the second pair must be within plus or minus scale_percentage of the first pair scale.
            # To better illustrate this, an example of this would be if the first_pair_scale is 3, then the second_pair_scale must be >= 1.5 as well as <= 4.5 in order for the agreement boolean to be true.
            change_in_scale_agreement = ((first_pair_scale - scale_percentage * first_pair_scale) <= second_pair_scale) and (second_pair_scale <= (first_pair_scale + scale_percentage * first_pair_scale))
           
            # If the change_in_scale_agreement is true and change in orientation is less than the specified orientation_degree, get the matched pair from the matched_pairs using pair_index (current iteration)
            # and add this matched pair to the consistent subset list.
            if (change_in_orientation < orientation_degree) and change_in_scale_agreement:                
                 consistent_subset.append(matched_pairs[pair_index])
                      
        
        # At the current RANSAC ITERATION, add the consistent_subset list computed when checking for consistency between the random matched pair and all matched pairs from the matched_pairs list
        # to the largest consistent subset list: consistent_subsets.
        consistent_subsets.append(consistent_subset)
        
    
   
    ######## NOTE Start: This part (till line 254) simply selects the largest subset in the array that consists all subsets through all RANSAC iterations (called consistent_subsets), and saves each element (matched pairs) of this subset 
    ########       into a final array called consistent_matched_pairs         
    
    # Initialize the index of consistent pairs which will be used to save the index of the subset with the most matched pairs found inside of the consistent_subsets (contains all consistent subsets across all RANSAC iterations)
    consistent_pairs_index = 0
    
    # Declare and initialize this variable which in the next step will save the largest subset (subset with most matched pairs) in the consistent_subsets (contains all consistent subsets across all RANSAC iterations)
    subset_length = 0
    
    # Iterate through every element in the consistent_subsets (contains all consistent subsets across all RANSAC iterations)
    for i in range(0, len(consistent_subsets)):
        
        # If the subset_length integer is smaller than the length of the subset in the current iteration, update subset_length to have the length of the current iteration's subset as well as 
        # update consistent_pairs_index to have the index of that subset in the array consistent_subset.
        # By the end of the entire for loop, subset_length will contain the length of the largest subset computed through all RANSAC iterations and
        # consistent_pairs_index will contain the index of this largest subset which will be used as the final array for matched pairs.
        if subset_length < len(consistent_subsets[i]):
            subset_length = len(consistent_subsets[i]) - 1
            consistent_pairs_index = i
            
    # Declare this new list which will save the final largest consistent subset across all RANSAC iterations.
    consistent_matched_pairs = []
    
    # Iterate through every element in the largest subset of consistent_subsets (contains all consistent subsets across all RANSAC iterations)
    # index consistent_pairs_index has the array position of the subset with the most matched pairs.
    # Add every element of this largest subset to the final list consistent_matched_pairs.
    for i in range(0, len(consistent_subsets[consistent_pairs_index])):         
        consistent_matched_pairs.append(consistent_subsets[consistent_pairs_index][i])
    ######## NOTE End.
        
    
    # END OF SECTION OF CODE TO REPLACE
    #
    im3 = DisplayMatches(im1, im2, consistent_matched_pairs)
    return im3

#Test run...
match('library','library2')

