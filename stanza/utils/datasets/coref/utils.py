from functools import lru_cache

class DynamicDepth():
    """
    Implements a cache + dynamic programming to find the relative depth of every word in a subphrase given the head word for every word.
    """
    def get_parse_depths(self, heads, start, end):
        """Return the relative depth for every word

        Args:
            heads (list): List where each entry is the index of that entry's head word in the dependency parse
            start (int): starting index of the heads for the subphrase
            end (int): ending index of the heads for the subphrase

        Returns:
            list: Relative depth in the dependency parse for every word
        """
        self.heads = heads[start:end]
        self.relative_heads = [h - start if h else -100 for h in self.heads] # -100 to deal with 'none' headwords

        depths = [self._get_depth_recursive(h) for h in range(len(self.relative_heads))]

        return depths

    @lru_cache(maxsize=None)
    def _get_depth_recursive(self, index):
        """Recursively get the depths of every index using a cache and recursion

        Args:
            index (int): Index of the word for which to calculate the relative depth

        Returns:
            int: Relative depth of the word at the index
        """
        # if the head for the current index is outside the scope, this index is a relative root
        if self.relative_heads[index] >= len(self.relative_heads) or self.relative_heads[index] < 0:
            return 0
        return self._get_depth_recursive(self.relative_heads[index]) + 1

def find_cconj_head(heads, upos, start, end):
    """
    Finds how far each word is from the head of a span, then uses the closest CCONJ to the head as the new head

    If no CCONJ is present, returns None
    """
    # use head information to extract parse depth
    dynamicDepth = DynamicDepth()
    depth = dynamicDepth.get_parse_depths(heads, start, end)
    depth_limit = 2

    # return first 'CCONJ' token above depth limit, if exists
    # unlike the original paper, we expect the parses to use UPOS, hence CCONJ instead of CC
    cc_indexes = [i for i in range(end - start) if upos[i+start] == 'CCONJ' and depth[i] < depth_limit]
    if cc_indexes:
        return cc_indexes[0] + start
    return None

