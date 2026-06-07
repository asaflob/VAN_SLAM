from PnP_RANSAC_ex2 import *
import pickle as pkl

class TrackingDatabase:
    def __init__(self):
        self.frame_to_tracks = {} #dict of track Ids
        self.track_to_frames = {} #dict of frame Ids
        self.feature_locations = {} #dict for (trackId, x_l, x_r, y)

    def get_tracks_by_frame(self, frame_id):
        """returns all the trackIds that appear on a given FrameId"""
        return self.frame_to_tracks.get(frame_id, [])

    def get_frames_by_track_id(self, track_id):
        """returns all the trackIds that appear on a given TrackId"""
        return self.track_to_frames.get(track_id, [])

    def get_feature_locations(self, track_id, frame_id):
        """returns the features locations of trackId on both left and right
        images as a (x,y,z)"""
        return self.feature_locations.get((track_id,frame_id), None)

    def add_feature(self, track_id, frame_id, xl, xr, y):
        """extends the database with a new track on a new frame"""
        if frame_id not in self.frame_to_tracks:
            self.frame_to_tracks[frame_id] = []
        if track_id not in self.frame_to_tracks[frame_id]:
            self.frame_to_tracks[frame_id].append(track_id)

        if track_id not in self.track_to_frames:
            self.track_to_frames[track_id] = []
        if frame_id not in self.track_to_frames[track_id]:
            self.track_to_frames[track_id].append(frame_id)

        self.feature_locations[(track_id, frame_id)] = (xl, xr, y)

    def serialize_the_tracking(self, frame_id, file_path):
        """serializes the tracking"""
        track_ids = self.get_tracks_by_frame(frame_id)

        frame_data={
            "frame_id" : frame_id,
            "features" : {}
        }
        for track_id in track_ids:
            frame_data["features"][track_id] = self.get_feature_locations(track_id, frame_id)

        with open(file_path, 'wb') as f:
            pkl.dump(frame_data, f)

    def load_frame_tracking(self, file_path):
        """reads tracking info of a specific frame from a file
        and adds it to the db"""

        with open(file_path, 'rb') as f:
            frame_data = pkl.load(f)

        frame_id = frame_data["frame_id"]

        for track_id, coords in frame_data["features"].items():
            if coords is not None:
                xl, xr, y = coords
                self.add_feature(track_id, frame_id, xl, xr, y)

    def save_entire_database(self, file_path):
        """Serializes the entire database to disk."""
        with open(file_path, 'wb') as f:
            pkl.dump(self, f)

    @staticmethod
    def load_entire_database(file_path):
        """Loads the entire database from disk."""
        with open(file_path, 'rb') as f:
            return pkl.load(f)

