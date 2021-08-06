import logging
import json

from pymongo import MongoClient, ASCENDING, DESCENDING, ReplaceOne, TEXT
from pymongo.errors import BulkWriteError, WriteError

logger = logging.getLogger(__name__)


class SuperFragment(object):

    def __init__(
            self,
            id,
            syn_ids=None,
            pre_partners=None,
            post_partners=None,
            ):

        self.id = id

        self.syn_ids = syn_ids if syn_ids is not None else []
        self.pre_partners = pre_partners if pre_partners is not None else []
        self.post_partners = post_partners if post_partners is not None else []

    def finalize(self):

        self.syn_ids = list(set(self.syn_ids))
        self.pre_partners = list(set(self.pre_partners))
        self.post_partners = list(set(self.post_partners))

    def to_json(self):

        return {
            'id': self.id,
            'syn_ids': self.syn_ids,
            'pre_partners': self.pre_partners,
            'post_partners': self.post_partners,
            }


class SuperFragmentDatabase(object):
    def __init__(
            self,
            db_name,
            db_host,
            db_col_name='superfragments',
            db_json=None,
            mode='r',
            segment_name_attribute='neuron_name',
            ):

        self.db_name = db_name
        self.db_host = db_host
        self.db_col = db_col_name
        self.mode = mode
        self.segment_name_attribute = segment_name_attribute

        self.__connect()
        self.__open_db()
        self.__open_collections()

        # self.client = MongoClient(host=db_host)
        # self.database = self.client[db_name]


        if mode == 'w':
            i = input("Drop collection %s.%s? Yes/[No] " % (db_name, db_col_name))
            if i == "Yes":
                self.database.drop_collection(self.superfragments)
                logger.info('Overwriting collection %s.%s' % (db_name, db_col_name))
            else:
                print("Aborted")
                exit(0)


        if db_col_name not in self.database.collection_names():

            self.superfragments.create_index(
                [
                    ('id', ASCENDING)
                ],
                name='id', unique=True)

    def write_superfragments(self, superfragments):
        if self.mode == 'r':
            raise RuntimeError("trying to write to read-only DB")

        if len(superfragments) == 0:
            logger.debug("No superfragments to write.")
            return

        if self.mode == 'r+':

            items = [sf.to_json() for sf in superfragments]

            try:
                self.__write(
                    self.superfragments,
                    ['id'],
                    items,
                    fail_if_exists=True
                    )

            except BulkWriteError as e:
                logger.error(e.details)
                raise

    def read_superfragments(self, sf_ids = None):
        if sf_ids is None:
            logger.debug("No ids provided, querying all superfragments in database")
            superfragments_dic = self.superfragments.find()
        elif sf_ids is not None:
            logger.debug("Querying superframents with following IDs:", sf_ids)
            
            superfragments_dic = self.superfragments.find(
                {
                    'id' : {'$in': sf_ids}
                })

        items = []
        for item in superfragments_dic:
            items.append(item)

        return items

    def __write(self, collection, match_fields, docs,
                fail_if_exists=False, fail_if_not_exists=False, delete=False):
        '''Writes documents to provided mongo collection, checking for restricitons.
        Args:
            collection (``pymongo.collection``):
                The collection to write the documents into.
            match_fields (``list`` of ``string``):
                The set of fields to match to be considered the same document.
            docs (``dict`` or ``bson``):
                The documents to insert into the collection
            fail_if_exists, fail_if_not_exists, delete (``bool``):
                see write_nodes or write_edges for explanations of these flags
            '''
        assert not delete, "Delete not implemented"
        match_docs = []
        for doc in docs:
            match_doc = {}
            for field in match_fields:
                match_doc[field] = doc[field]
            match_docs.append(match_doc)

        if fail_if_exists:
            self.__write_fail_if_exists(collection, match_docs, docs)
        elif fail_if_not_exists:
            self.__write_fail_if_not_exists(collection, match_docs, docs)
        else:
            self.__write_no_flags(collection, match_docs, docs)

    def __write_no_flags(self, collection, old_docs, new_docs):
        bulk_query = [ReplaceOne(old, new, upsert=True)
                      for old, new in zip(old_docs, new_docs)]
        collection.bulk_write(bulk_query)

    def __write_fail_if_exists(self, collection, old_docs, new_docs):
        for old in old_docs:
            if collection.count_documents(old):
                raise WriteError(
                        "Found existing doc %s and fail_if_exists set to True."
                        " Aborting write for all docs." % old)
        collection.insert_many(new_docs)

    def __write_fail_if_not_exists(self, collection, old_docs, new_docs):
        for old in old_docs:
            if not collection.count_documents(old):
                raise WriteError(
                        "Did not find existing doc %s and fail_if_not_exists "
                        "set to True. Aborting write for all docs." % old)
        bulk_query = [ReplaceOne(old, new, upsert=False)
                      for old, new in zip(old_docs, new_docs)]
        result = collection.bulk_write(bulk_query)
        assert len(new_docs) == result.matched_count,\
            ("Supposed to replace %s docs, but only replaced %s"
                % (len(new_docs), result.matched_count))

    def __connect(self):
        '''Connects to Mongo client'''
        self.client = MongoClient(self.db_host) 

    def __open_db(self):
        '''Opens Mongo database'''
        self.database = self.client[self.db_name]

    def __open_collections(self):
        '''Opens the node, edge, and meta collections'''

        self.superfragments = self.database[self.db_col]

    def __disconnect(self):
        '''Closes the mongo client and removes references
        to all collections and databases'''
        self.database = None
        self.client.close()
        self.client = None

    def close(self):
        self.__disconnect()
