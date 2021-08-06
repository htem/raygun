import logging
import json

from pymongo import MongoClient, ASCENDING, TEXT

logger = logging.getLogger(__name__)


class SynapseDatabase(object):
    """" Database interface for synapses. One document corresponds to one synapse"""

    def __init__(
            self,
            db_name,
            db_host,
            db_col_name='synapses',
            # db_json=None,
            mode='r+'):
        # if db_json is not None:
        #     assert db_name is None, 'both db_name and db_json provided, unclear what to do'
        #     with open(db_json) as f:
        #         db_config = json.load(f)
        #     db_name = db_config['db_name']
        #     db_host = db_config['db_host']
        #     db_col_name = db_config['db_col']
        self.db_name = db_name
        self.db_host = db_host

        self.synapses_collection_name = db_col_name
        # self.db_col = db_col_name
        self.mode = mode
        self.client = MongoClient(host=db_host)
        self.database = self.client[db_name]

        # open collections
        self.synapses = self.database[db_col_name]
        self.__connect()
        self.__open_db()
        self.__open_collections()

        if mode == 'w':

            i = input("Drop collection %s.%s? Yes/[No] " % (db_name, db_col_name))
            if i == "Yes":
                self.database.drop_collection(self.synapses)
                logger.info('Overwriting collection %s.%s' % (db_name, db_col_name))
            else:
                print("Aborted")
                exit(0)

        self.__create_synapses_collection()

    def write_synapses(self, synapses):

        if self.mode == 'r':
            raise RuntimeError("trying to write to read-only DB")

        if len(synapses) == 0:
            logger.debug("No synapse to write.")
            return

        db_list = []
        
        for syn in synapses:
            syn_dic = {
                'id': int(syn.id),
                # adding the index location
                'x': int(syn.zyx[2]),
                'y': int(syn.zyx[1]),
                'z': int(syn.zyx[0]),
                'pre_z': int(syn.location_pre[0]),
                'pre_y': int(syn.location_pre[1]),
                'pre_x': int(syn.location_pre[2]),
                'post_z': int(syn.location_post[0]),
                'post_y': int(syn.location_post[1]),
                'post_x': int(syn.location_post[2]),
                # 'proofread': False,
                # 'pr_false_positive': False,
                # 'pr_user_id': '',
            }
            if syn.score is not None:
                syn_dic['score'] = float(syn.score)
            if syn.area is not None:
                syn_dic['area'] = int(syn.area)
            if syn.id_superfrag_pre is not None:
                syn_dic['id_superfrag_pre'] = int(syn.id_superfrag_pre)
            if syn.id_superfrag_post is not None:
                syn_dic['id_superfrag_post'] = int(syn.id_superfrag_post)

            db_list.append(syn_dic)

            # print("Inserting synapse: ID", syn_dic['id'])
            # self.synapses.insert_one(syn_dic)
        
        # print("Inserting synapses Ids: ", syn_dic_id)
        self.synapses.insert_many(db_list)
        
        logger.debug("Insert %d synapses" % len(synapses))


    def read_synapses(self, roi=None):
        """ Read synapses from database.

        Args:
            roi (``daisy.Roi``, optional):
                If given, restrict reading synapses to ROI. If not given, all synapses are read.
        Returns:
            ``list`` of ``dic``: List of synapses in dictionary format.
        """

        if roi is None:
            logger.debug("No roi provided, querying all synapses in database")
            synapses_dic = self.synapses.find({})
        elif roi is not None:
            logger.debug("Querying synapses in %s", roi)
            bz, by, bx = roi.get_begin()
            ez, ey, ex = roi.get_end()
            synapses_dic = self.synapses.find(
                {
                    'z': {'$gte': bz, '$lt': ez},
                    'y': {'$gte': by, '$lt': ey},
                    'x': {'$gte': bx, '$lt': ex}
                })

        return synapses_dic

    def __create_synapses_collection(self):

        indexes = self.synapses.index_information()

        if 'zyx_position' not in indexes:
            self.synapses.create_index(
                [
                    ('z', ASCENDING),
                    ('y', ASCENDING),
                    ('x', ASCENDING)

                ],
                name='zyx_position')

        if 'id' not in indexes:
            self.synapses.create_index(
                [
                    ('id', ASCENDING)
                ],
                name='id', unique=True)

        if 'proofread' not in indexes:
            self.synapses.create_index(
                [
                    ('proofread', ASCENDING)
                ],
                name='proofread')

        if 'pr_false_positive' not in indexes:
            self.synapses.create_index(
                [
                    ('pr_false_positive', ASCENDING)
                ],
                name='pr_false_positive')

    def __connect(self):
        '''Connects to Mongo client'''
        self.client = MongoClient(self.db_host)

    def __open_db(self):
        '''Opens Mongo database'''
        self.database = self.client[self.db_name]

    def __open_collections(self):
        '''Opens the node, edge, and meta collections'''
        self.synapses = self.database[self.synapses_collection_name]

