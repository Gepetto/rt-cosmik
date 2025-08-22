# pip install rosbags
from pathlib import Path
from rosbags.highlevel import AnyReader

bagpath = Path("/home/msabbah/pinocchio-3x/src/rt-cosmik/output/Alessandro/Alessandro_sanding.bag")       # or a ros2 bag directory or .mcap
with AnyReader([bagpath]) as reader:
    # List topics and types
    for c in reader.connections:
        print(c.topic, c.msgtype)

    # Read a few messages from all topics
    for i, (conn, t, raw) in enumerate(reader.messages()):
        msg = reader.deserialize(raw, conn.msgtype)
        print(conn.topic, type(msg).__name__, t)
        if i > 20:
            break

    # Example: filter a specific topic and access fields
    joints = [c for c in reader.connections if c.topic == '/joint_states']
    for conn, t, raw in reader.messages(connections=joints):
        msg = reader.deserialize(raw, conn.msgtype)
        print(t, list(zip(msg.name, msg.position)))
        break
