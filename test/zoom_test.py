import cv2
from tqdm import tqdm

def crop_and_align_videos(dji_path, mapir_path, output_overlay, output_dji, scale_factor, dji_sync_sec=0.0, mapir_sync_sec=0.0):
    # Open both synchronized video files
    cap_dji = cv2.VideoCapture(dji_path)
    cap_mapir = cv2.VideoCapture(mapir_path)

    if not cap_dji.isOpened() or not cap_mapir.isOpened():
        print("Error: Could not open one or both video files.")
        return

    # Get dimensions and FPS
    dji_w = int(cap_dji.get(cv2.CAP_PROP_FRAME_WIDTH))
    dji_h = int(cap_dji.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dji_fps = cap_dji.get(cv2.CAP_PROP_FPS)
    
    mapir_w = int(cap_mapir.get(cv2.CAP_PROP_FRAME_WIDTH))
    mapir_h = int(cap_mapir.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mapir_fps = cap_mapir.get(cv2.CAP_PROP_FPS)

    print(f"DJI FPS: {dji_fps:.2f} | MAPIR FPS: {mapir_fps:.2f}")

    # Calculate crops
    crop_w = int(dji_w / scale_factor)
    crop_h = int(dji_h / scale_factor)
    x_offset = (dji_w - crop_w) // 2
    y_offset = (dji_h - crop_h) // 2

    # Set up video writers (using MAPIR framerate as the base)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_overlay = cv2.VideoWriter(output_overlay, fourcc, mapir_fps, (mapir_w, mapir_h))
    out_dji_only = cv2.VideoWriter(output_dji, fourcc, mapir_fps, (mapir_w, mapir_h))

    # Seek MAPIR to its sync point
    mapir_start_frame = int(mapir_sync_sec * mapir_fps)
    cap_mapir.set(cv2.CAP_PROP_POS_FRAMES, mapir_start_frame)
    
    # Seek DJI to its sync point
    current_dji_frame = int(dji_sync_sec * dji_fps)
    cap_dji.set(cv2.CAP_PROP_POS_FRAMES, current_dji_frame)
    ret_dji, frame_dji = cap_dji.read() # Read initial DJI frame

    frame_count = 0
    show_preview = True
    
    total_frames = int(cap_mapir.get(cv2.CAP_PROP_FRAME_COUNT)) - mapir_start_frame
    pbar = tqdm(total=total_frames, desc="Aligning Frames")

    while True:
        ret_mapir, frame_mapir = cap_mapir.read()
        if not ret_mapir or not ret_dji:
            break
            
        # Calculate what time we are at relative to the sync point
        mapir_elapsed_sec = frame_count / mapir_fps
        expected_dji_sec = dji_sync_sec + mapir_elapsed_sec
        expected_dji_frame = int(expected_dji_sec * dji_fps)
        
        # Fast-forward DJI video until it catches up to the expected time
        while current_dji_frame < expected_dji_frame:
            ret_dji, frame_dji = cap_dji.read()
            current_dji_frame += 1
            if not ret_dji:
                break
                
        if not ret_dji:
            break

        # Apply center crop to the DJI frame via NumPy slicing
        cropped_dji = frame_dji[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w]
        resized_dji = cv2.resize(cropped_dji, (mapir_w, mapir_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Save just the DJI crop
        out_dji_only.write(resized_dji)

        # Blend the two frames (50% DJI, 50% MAPIR) to check alignment
        blended_frame = cv2.addWeighted(resized_dji, 0.5, frame_mapir, 0.5, 0)
        out_overlay.write(blended_frame)
        
        if show_preview:
            preview_frame = cv2.resize(blended_frame, (mapir_w * 2, mapir_h * 2), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Alignment Preview (Press Q to close preview)', preview_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                show_preview = False
                cv2.destroyAllWindows()
                print("Preview closed. Processing the rest of the video in the background... please wait.")

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap_dji.release()
    cap_mapir.release()
    out_overlay.release()
    out_dji_only.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames.")
    print(f"1. Overlay saved to: {output_overlay}")
    print(f"2. Cropped DJI saved to: {output_dji}")

# --- Execution ---
DJI_VIDEO = r'D:\Chirayu\test\rgb.MP4'
MAPIR_VIDEO = r'D:\Chirayu\test\nir.MP4'
OUTPUT_OVERLAY = r'D:\Chirayu\test\aligned_overlay.mp4'
OUTPUT_DJI_ONLY = r'D:\Chirayu\test\rgb_cropped_synced.mp4'

ESTIMATED_SCALE_FACTOR = 1.8 

# Timestamp in seconds where you hear the clap
DJI_SYNC_TIME_SEC = 1
MAPIR_SYNC_TIME_SEC = 5

crop_and_align_videos(DJI_VIDEO, MAPIR_VIDEO, OUTPUT_OVERLAY, OUTPUT_DJI_ONLY, ESTIMATED_SCALE_FACTOR, DJI_SYNC_TIME_SEC, MAPIR_SYNC_TIME_SEC)