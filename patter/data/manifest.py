import json


class Manifest(object):
    def __init__(self, manifest_filepath, max_duration=None, min_duration=None):
        ids = []
        duration = 0.0
        filtered_duration = 0.0
        with open(manifest_filepath) as fh:
            for line in fh:
                data = json.loads(line)
                if min_duration is not None and data['duration'] < min_duration:
                    filtered_duration += data['duration']
                    continue
                if max_duration is not None and data['duration'] > max_duration:
                    filtered_duration += data['duration']
                    continue
                ids.append(data)
                duration += data['duration']
        self._data = ids
        self._size = len(ids)
        self._duration = duration
        self._filtered_duration = filtered_duration

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return self._size

    @property
    def duration(self):
        return self._duration

    @property
    def filtered_duration(self):
        return self._filtered_duration
