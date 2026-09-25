#!/usr/bin/env python3
"""
Play a RealSense Viewer clip (.db3) as if the camera were plugged in.

Test helper only, not used on the device. It sends each depth picture from the clip on
the same channel, with the same message type and pixel format, as the live RealSense
driver, at the speed it was recorded.

Usage (inside Ubuntu, with ROS 2 switched on):
  python3 tools/play_clip.py CLIP                       play once
  python3 tools/play_clip.py CLIP --loop                play forever (Ctrl+C to stop)
  python3 tools/play_clip.py CLIP --blank-from 2 --blank-to 5
                                                        seconds 2 to 5 become all zeros,
                                                        like a hand over the lens
"""

import argparse
import array
import pathlib
import sqlite3
import time

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

try:                                    # Ubuntu 22.04 package: python3-zstd
    import zstd

    def unpack(blob):
        return zstd.decompress(blob)
except ImportError:                     # newer Ubuntu package: python3-zstandard
    import zstandard

    def unpack(blob):
        return zstandard.ZstdDecompressor().decompressobj().decompress(blob)

CLIP_DEPTH_CHANNEL = '/device_0/sensor_0/Depth_0/image/data'   # RealSense Viewer's name
LIVE_DEPTH_CHANNEL = '/camera/camera/depth/image_rect_raw'     # RealSense ROS driver's name
ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'                                # start of a compressed blob


def open_clip(path):
    """Open the clip read-only and return (connection, list of (message id, timestamp ns))."""
    uri = pathlib.Path(path).resolve().as_uri() + '?immutable=1'
    con = sqlite3.connect(uri, uri=True)
    frames = con.execute(
        'SELECT m.id, m.timestamp FROM messages m JOIN topics t ON t.id = m.topic_id '
        'WHERE t.name = ? ORDER BY m.timestamp', (CLIP_DEPTH_CHANNEL,)).fetchall()
    if not frames:
        raise SystemExit(f'No depth pictures found in {path}')
    return con, frames


def read_picture(con, message_id):
    """Read one depth picture from the clip and turn it into a ROS Image message."""
    blob = con.execute('SELECT data FROM messages WHERE id = ?', (message_id,)).fetchone()[0]
    if blob[:4] == ZSTD_MAGIC:
        blob = unpack(blob)
    msg = deserialize_message(blob, Image)
    msg.encoding = '16UC1'                          # what the live driver sends (same bytes)
    msg.header.frame_id = 'camera_depth_optical_frame'
    return msg


def main():
    parser = argparse.ArgumentParser(description='Play a RealSense .db3 clip as a live camera.')
    parser.add_argument('clip', help='path to the clip file')
    parser.add_argument('--topic', default=LIVE_DEPTH_CHANNEL, help='channel to send on')
    parser.add_argument('--loop', action='store_true', help='play again and again')
    parser.add_argument('--speed', type=float, default=1.0, help='2 = twice as fast')
    parser.add_argument('--blank-from', type=float, default=None, help='seconds into the clip')
    parser.add_argument('--blank-to', type=float, default=None, help='seconds into the clip')
    args = parser.parse_args()

    con, frames = open_clip(args.clip)
    length_s = (frames[-1][1] - frames[0][1]) / 1e9
    print(f'{len(frames)} depth pictures, {length_s:.1f} s, sending on {args.topic}')

    rclpy.init()
    node = Node('play_clip')
    pub = node.create_publisher(Image, args.topic, 5)
    time.sleep(1.0)                                 # give listeners a moment to connect

    try:
        while True:
            start = time.monotonic()
            for number, (message_id, stamp_ns) in enumerate(frames, start=1):
                clip_s = (stamp_ns - frames[0][1]) / 1e9
                wait = start + clip_s / args.speed - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                msg = read_picture(con, message_id)
                blank = (args.blank_from is not None and args.blank_to is not None
                         and args.blank_from <= clip_s <= args.blank_to)
                if blank:
                    msg.data = array.array('B', bytes(len(msg.data)))  # all 0 = "no reading"
                if not rclpy.ok():                      # we were asked to stop
                    return
                msg.header.stamp = node.get_clock().now().to_msg()
                pub.publish(msg)
                if number % 30 == 0 or number == len(frames):
                    note = '  (blanked)' if blank else ''
                    print(f'  {clip_s:5.1f} s  picture {number}/{len(frames)}{note}')
            if not args.loop:
                break
    except (KeyboardInterrupt, Exception):          # stopped mid-send: not an error
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
