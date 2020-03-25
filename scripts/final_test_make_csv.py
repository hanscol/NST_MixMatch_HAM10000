import os

image_dir = '/scratch/hansencb/ISIC2018_Task3_Test_Input'
out_csv = 'submit_test.csv'
with open(out_csv, 'w') as f:
    for image in os.listdir(image_dir):
        if image.endswith('.jpg'):
            path = os.path.join(image_dir, image)
            f.write('{},{}\n'.format(path, '0'))
