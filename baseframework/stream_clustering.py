"""
Class modeling the Online Phase behavior
"""
import numpy as np


class OnlinePhase_StreamClusterer:
    def __init__(self):
        pass

    """
        Updates Stream Clusterer with a single instance at a certain timestamp
        
        Parameters
        ----
        x: array-like, instance inserted
        t: int, timestamp when instance is inserted, optional, increment by 1 if None is provided
        
        Returns
        ----
        Tuple of summary assignment id, cluster id, event_log string
    """
    def learn_one(self, x, t: int) -> tuple[int, int, str]:
        pass

    """
        Updates Stream Clusterer with multiple instances each at a certain timestamp

        Parameters
        ----
        xs: list of array-likes; the inserted instances 
        ts: list of ints; timestamps when instance is inserted

        Returns
        ----
        List containing the results from learn_one
    """
    def learn(self, xs, ts) -> list[tuple[int, int, str]]:
        outputs = []
        for i in range(len(xs)):
            outputs.append(self.learn_one(xs[i], ts[i]))
        return outputs

    """
        Returns the current state of the summarization of the algorithm
        
        Returns
        ----
        Dictionary mapping the summary assignment id to a representation of the summary structure (typically: dictionary with position, extent/radius, weight) 
    """

    def get_summary(self) -> dict:
        pass

"""
Class modeling the Offline Phase behavior
"""

class OfflinePhase_Clusterer:
    def __init__(self):
        pass


    """
        Predict the clusters for the summaries
    
        Parameters
        ----
        summaries: dict
            dictionary of summaries (output of get_summary)
        
        Returns
        ----
            cluster labeling for the summaries, mapping summary assignment id to its cluster label (noise labels is -1)
    """

    def predict_summaries(self, summaries: dict) -> dict:
        pass

    """
        Predict the clusters based on the assignment of instances to the summaries
        Parameters
        ----
        assignments: list[int]
            summary assignment id for instances to cluster; 
            either output from learn_one/learn or post-summary extraction matching through nearest neighbor/grid cell assignment
        
        summaries: dict
            dictionary of summaries (output of get_summary)
        
        Returns
        ----
            cluster labeling for the summaries, mapping summary assignment id to its cluster label (noise labels is -1)
    """

    def predict_assignment(self, assignments: list[int], summaries: dict) -> list[int]:
        summary_clustering = self.predict_summaries(summaries)
        summary_clustering[-1] = -1
        return [summary_clustering[a] for a in assignments]


