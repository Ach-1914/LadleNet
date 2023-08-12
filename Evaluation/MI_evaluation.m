function mutual_informationR = mutual_information(image1, image2, grey_level)

grey_matrix1 = uint8(image1 * (grey_level - 1));
grey_matrix2 = uint8(image2 * (grey_level - 1));

H1 = entropy(grey_matrix1);
H2 = entropy(grey_matrix2);

H12 = Hab(grey_matrix1, grey_matrix2, grey_level);

mutual_informationR = H1 + H2 - H12;
