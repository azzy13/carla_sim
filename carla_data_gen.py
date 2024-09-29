import carla
import math
import random
import time
import queue
import numpy as np
import cv2
import pygame
from PIL import Image
import json

def main():
    # Connect to the client and get the world object
    client = carla.Client('localhost', 2000) 
    client.set_timeout(200.0)
    world  = client.reload_world()
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points() 

    # Get the blueprint for the vehicle you want
    vehicle_bp = bp_lib.find('vehicle.nissan.patrol_2021') 

    # Try spawning the vehicle at a randomly chosen spawn point
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    spectator = world.get_spectator()

    # Spawn vehicles
    for i in range(50): 
        vehicle_bp = random.choice(bp_lib.filter('vehicle')) 
        npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points)) 

    # Set up camera on spectator (drone)
    location = carla.Location(x=14.0, y=29.0, z=6.0)
    rotation = carla.Rotation(pitch=0.0, yaw=math.pi/2, roll=0.0)
    transform = carla.Transform(location, rotation)
    print(location, rotation)

    camera_drone = bp_lib.find('sensor.camera.rgb')
    camera_drone.set_attribute('image_size_x', '1920')
    camera_drone.set_attribute('image_size_y', '1080')
    camera_drone.set_attribute('sensor_tick', '1.0')
    camera_drone.set_attribute('fov', '110')
    camera = world.spawn_actor(camera_drone, transform)

    #To attach a seg_camera on drone
    segmentation_camera = setup_instance_segmentation_camera(world, bp_lib, transform)

    #To attach a lidar on drone
    lidar = setup_semantic_lidar(world, bp_lib, transform)

    # Get the world to camera matrix
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    # Get the attributes from the camera
    image_w = camera_drone.get_attribute("image_size_x").as_int()
    image_h = camera_drone.get_attribute("image_size_y").as_int()
    fov = camera_drone.get_attribute("fov").as_float()

    # Calculate the camera projection matrix to project from 3D -> 2D
    K = build_projection_matrix(image_w, image_h, fov)
    K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

    image_queue = queue.Queue()
    camera.listen(image_queue.put)
    width = 1920
    height = 1080
    pygame.init()
    window = pygame.display.set_mode(size=(width,height))
    font = pygame.font.SysFont(None, 24)
    video_surf = None
    simulation_dataset = {}
    simulation_dataset["images"] = []
    simulation_dataset["annotations"] = []

    #render 2D bboxes
    running = True
    tick = 0
    max_tick = 12000
    while running:
        tick = tick + 1
        print(tick)
        if (tick > max_tick):
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # Retrieve and reshape the image
        world.tick()
        image = None
        try:
            image = image_queue.get(timeout=1)
            print("Received image!")
        except:
            print("No image!")
            continue

        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        img_filename = 'dataset/stationary_camera/rgb/%06d.png' % image.frame
        img_bgr = img[:,:, :3]
        #print(img_bg)
        img_rgb = img_bgr[:,:, ::-1]
        img_PIL = Image.fromarray(img_rgb)
        img_PIL.save(img_filename)
        simulation_dataset["images"].append({
            "file_name": img_filename,
            "height": 1920,
            "width": 1080,
            "id": image.frame
        })

        # Get the camera matrix 
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        for npc in world.get_actors().filter('*vehicle*'):

            # Filter out the ego vehicle
            if npc.id != vehicle.id:

                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(transform.location)

                # Filter for the vehicles within 50m
                if dist < 70:

                # Calculate the dot product between the forward vector
                # of the vehicle and the vector between the vehicle
                # and the other vehicle. We threshold this dot product
                # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                    forward_vec = transform.get_forward_vector()
                    ray = npc.get_transform().location - transform.location

                    if forward_vec.dot(ray) > 0:
                        p1 = get_image_point(bb.location, K, world_2_camera)
                        verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                        x_max = -10000
                        x_min = 10000
                        y_max = -10000
                        y_min = 10000

                        for vert in verts:
                            p = get_image_point(vert, K, world_2_camera)
                            # Find the rightmost vertex
                            if p[0] > x_max:
                                x_max = p[0]
                            # Find the leftmost vertex
                            if p[0] < x_min:
                                x_min = p[0]
                            # Find the highest vertex
                            if p[1] > y_max:
                                y_max = p[1]
                            # Find the lowest  vertex
                            if p[1] < y_min:
                                y_min = p[1]

                        cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
                        cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                        cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
                        cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                        bbox = [x_min, y_min, x_max, y_max]
                        # Draw vertices
                        vertices = [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
                        # for vertex in vertices:
                        #     cv2.circle(img, vertex, 5, (0, 255, 0), -1)  # Draw a circle at each vertex

                        # Put the text for each vertex
                        # for vertex in vertices:
                        #     cv2.putText(img, str(vertex), (vertex[0] + 10, vertex[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # Convert bounding box coordinates to string
                        bbox = [x_min, y_min, x_max, y_max]
                        bbox_str = str(bbox)

                        # Put the bounding box string on the image
                        # cv2.putText(img, bbox_str, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        
                        rounded_bbox = [round(value) for value in bbox]
                        print(rounded_bbox)
                        simulation_dataset["annotations"].append({
                            "image_id": image.frame,
                            "bbox": rounded_bbox
                        })

        video_surf = pygame.image.frombuffer(img, (image.width, image.height), "BGRA")
        window.blit(video_surf, (0,0))
        pygame.display.flip()
        #pygame.time.wait(5)
    pygame.quit()

    with open('simulation_data.json', 'w') as json_file:
        json.dump(simulation_dataset, json_file)
        

if __name__ == "__main__":
    main()

#Camera geometric projection -> 3D points
def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

#3D to 2D
def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

def setup_instance_segmentation_camera(world, bp_lib, transform):
    seg_camera = bp_lib.find('sensor.camera.instance_segmentation')
    seg_camera.set_attribute('image_size_x', '1920')
    seg_camera.set_attribute('image_size_y', '1080')
    seg_camera.set_attribute('sensor_tick', '1.0')
    seg_camera.set_attribute('fov', '110')
    segmentation_camera = world.spawn_actor(seg_camera, transform)
    return segmentation_camera

def setup_semantic_lidar(world, bp_lib, transform):
    sem_lidar = bp_lib.find('sensor.lidar.ray_cast_semantic')
    sem_lidar.set_attribute('sensor_tick', '1.0')
    lidar = world.spawn_actor(sem_lidar, transform)
    return lidar