# create_homography.py
import cv2, json
from pathlib import Path

def main(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed. Check camera index or connection.")
        cap.release()
        return

    points = []
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([int(x), int(y)])
            print("Point:", x, y)

    cv2.namedWindow("click")
    cv2.setMouseCallback("click", click)

    print("Click exactly 4 points (corners of exam area) in order: top-left, top-right, bottom-right, bottom-left.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        disp = frame.copy()
        for p in points:
            cv2.circle(disp, tuple(p), 6, (0,255,0), -1)
        cv2.imshow("click", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if len(points) == 4:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(points) != 4:
        print("Need 4 points. Exiting.")
        return

    rows = int(input("Enter number of rows (e.g., 6): ").strip())
    cols = int(input("Enter number of cols (e.g., 8): ").strip())
    print("Now enter roll numbers for each row separated by spaces. Example: 1001 1002 ...")
    rolls = []
    for r in range(rows):
        line = input(f"Row {r+1}: ").strip().split()
        if len(line) != cols:
            print(f"Expected {cols} roll numbers, got {len(line)}. Exiting.")
            return
        rolls.append(line)

    dst_w = cols * 100
    dst_h = rows * 100
    dst = [[0,0],[dst_w,0],[dst_w,dst_h],[0,dst_h]]

    cfg = {
        "rows": rows,
        "cols": cols,
        "rolls": rolls,
        "homography_src": points,
        "homography_dst": dst
    }

    Path("seating_map.json").write_text(json.dumps(cfg, indent=2))
    print("Saved seating_map.json. You can edit it later if needed.")

if __name__ == "__main__":
    import sys
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(idx)
