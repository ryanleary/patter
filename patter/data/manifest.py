import json


class Manifest(object):
    def __init__(self, manifest_path, max_duration=None, min_duration=None, sort_by_duration=True):
        ids = []
        duration = 0.0
        filtered_duration = 0.0
        with open(manifest_path) as fh:
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
        if sort_by_duration:
            ids = sorted(ids, key=lambda x: x['duration'])
        self._data = ids
        self._size = len(ids)
        self._duration = duration
        self._filtered_duration = filtered_duration

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter(self._data)

    @property
    def duration(self):
        return self._duration

    @property
    def filtered_duration(self):
        return self._filtered_duration

    @property
    def data(self):
        return list(self._data)
