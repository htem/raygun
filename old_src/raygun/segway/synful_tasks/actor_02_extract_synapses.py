import copy
import json
import os
import logging
import numpy as np
import sys
import daisy
import pymongo
import time
# import math
import synapse
import detection
from database_synapses import SynapseDatabase
from database_superfragments import SuperFragmentDatabase, SuperFragment
from daisy import Coordinate


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def __create_unique_syn_id(zyx):

    id = 0
    binary_str = []
    for i in zyx:
        binary_str.append("{0:021b}".format(int(i)))
    id = int(''.join(binary_str), 2)

    return id


def __create_syn_ids(zyx_list):

    ret = []
    for zyx in zyx_list:
        ret.append(__create_unique_syn_id(zyx))

    assert(len(set(ret)) == len(ret))  # make sure that we created unique IDs
    return ret


def __create_syn_locations(predicted_syns, target_sites):

    if len(predicted_syns) != len(target_sites):
        print("ERROR: pre and post synaptic site do not have same length!")
        print("Synapses location was not created!")

    else:
        loc_zyx =[]
        for i in range(len(predicted_syns)):
            loc_zyx.append((predicted_syns[i]+target_sites[i])/2)

        return loc_zyx
           
def extract_synapses(ind_pred_ds,
                     dir_pred_ds,
                     segment_ds,
                     parameters,
                     block,
                     prediction_post_to_pre=True,
                     ):
    """Extract synapses from the block and write in the DB"""
    ##### EXTRACT SYNAPSES
    start_time = time.time()

    pred_roi = ind_pred_ds.intersect(block.read_roi).roi
    logger.debug('reading roi %s' % pred_roi)
    zchannel = ind_pred_ds.to_ndarray(roi=pred_roi)

    if len(zchannel.shape) == 4:
        zchannel = np.squeeze(zchannel[3, :])
    if zchannel.dtype == np.uint8:
        zchannel = zchannel.astype(np.float32)
        zchannel /= 255.  # Convert to float
        logger.debug('Rescaling z channel with 255')

    voxel_size = np.array(ind_pred_ds.voxel_size)
    predicted_syns, scores, areas = detection.find_locations(zchannel, parameters,
                                                      voxel_size)  # In world units.

    # Filter synapses for scores.
    new_scorelist = []
    new_arealist = []
    if parameters.score_thr is not None:
        filtered_list = []
        for ii, loc in enumerate(predicted_syns):
            score = scores[ii]
            area = areas[ii]
            if score > parameters.score_thr:
                filtered_list.append(loc)
                new_scorelist.append(score)
                new_arealist.append(area)

        logger.info(
            'filtered out %i out of %d' % (len(predicted_syns) - len(filtered_list), len(predicted_syns)))
        predicted_syns = filtered_list
        scores = new_scorelist
        areas = new_arealist

    # Load direction vectors and find target location
    dirmap = dir_pred_ds.to_ndarray(roi=pred_roi)

    # Before rescaling, convert back to float
    dirmap = dirmap.astype(np.float32)
    if 'scale' in dir_pred_ds.data.attrs:
        scale = dir_pred_ds.data.attrs['scale']
        dirmap = dirmap * 1. / scale
    else:
        logger.warning(
            'Scale attribute of dir vectors not set. Assuming dir vectors unit: nm, max value {}'.format(
                np.max(dirmap)))

    target_sites = detection.find_targets(predicted_syns, dirmap,
                                          voxel_size=voxel_size)
    # Synapses need to be shifted to the global ROI
    # (currently aligned with block.roi)
    for loc in predicted_syns:
        loc += np.array(pred_roi.get_begin())
    for loc in target_sites:
        loc += np.array(pred_roi.get_begin())

    # Filter post synaptic location not incuded in the block: do not compute syynapses
    post_syns = []
    filt_ind = []
    for i, (pred_loc, target_loc) in enumerate(zip(predicted_syns, target_sites)):
        if block.write_roi.contains(pred_loc) and \
                pred_roi.contains(target_loc):
            # note: ideally target_loc should have been contained in read_roi
            # (if not, context should be increased)
            # but corner cases where the block_roi is at the end of the volume
            # can give rise to errors
            post_syns.append(pred_loc)
            filt_ind.append(i)

    pre_syns = list(np.array(target_sites)[filt_ind])

    assert(prediction_post_to_pre)
    if not prediction_post_to_pre:
        pre_syns_tmp = pre_syns
        pre_syns = copy.deepcopy(post_syns)
        post_syns = copy.deepcopy(pre_syns_tmp)

    segment_ds = segment_ds[pred_roi]
    segment_ds.materialize()

    # Superfragments IDs
    ids_sf_pre = []
    for pre_syn in pre_syns:
        pre_syn = Coordinate(pre_syn)
        pre_super_fragment_id = segment_ds[pre_syn]
        assert pre_super_fragment_id is not None
        ids_sf_pre.append(pre_super_fragment_id)
    # print("Pre super fragment ID: ", ids_sf_pre)

    ids_sf_post = []
    for post_syn in post_syns:
        post_syn = Coordinate(post_syn)
        post_super_fragment_id = segment_ds[post_syn]
        assert post_super_fragment_id is not None
        ids_sf_post.append(post_super_fragment_id)
    # print("Post super fragment ID: ", ids_sf_post)

    # filter false positives
    pre_syns_f = []
    post_syns_f = []
    scores_f = []
    areas_f = []
    i_f = [] # indices to consider
    for i in range(len(ids_sf_pre)):
        if ids_sf_pre[i] == ids_sf_post[i]:
            continue
        if ids_sf_pre[i] == 0 or ids_sf_post[i] == 0:
            continue
        pre_syns_f.append(pre_syns[i])
        post_syns_f.append(post_syns[i])
        scores_f.append(scores[i])
        areas_f.append(areas[i])
        i_f.append(i)

    ids_sf_pre = list(np.array(ids_sf_pre)[i_f])
    ids_sf_post = list(np.array(ids_sf_post)[i_f])
    # Create xyz locations
    zyx = __create_syn_locations(pre_syns_f, post_syns_f)
    # Create IDs for synpses from volume coordinates
    # ids = __create_syn_ids(zyx)
    ids = __create_syn_ids(post_syns_f)  # make ID based on post for uniqueness
    # print("Synapses IDs: ", ids)

    synapses = synapse.create_synapses(pre_syns_f, post_syns_f,
                                   scores=scores_f, areas=areas_f, ID=ids, zyx=zyx,
                                   ids_sf_pre=ids_sf_pre,
                                   ids_sf_post=ids_sf_post)

    # print("Extraction synapses execution time = %f s" % (time.time() - start_time))

    return synapses 


def extract_superfragments(synapses, write_roi):

    superfragments = {}
    for syn in synapses:

        pre_partner_id = int(syn.id_superfrag_pre)
        post_partner_id = int(syn.id_superfrag_post)

        if write_roi.contains(Coordinate(syn.location_pre)):

            if pre_partner_id not in superfragments:
                superfragments[pre_partner_id] = SuperFragment(id=pre_partner_id)

            superfragments[pre_partner_id].syn_ids.append(syn.id)
            superfragments[pre_partner_id].post_partners.append(post_partner_id)

        if write_roi.contains(Coordinate(syn.location_post)):

            if post_partner_id not in superfragments:
                superfragments[post_partner_id] = SuperFragment(id=post_partner_id)

            superfragments[post_partner_id].syn_ids.append(syn.id)
            superfragments[post_partner_id].pre_partners.append(pre_partner_id)

    superfragments_list = [superfragments[item] for item in superfragments]
    for sf in superfragments_list:
        sf.finalize()

    return superfragments_list


if __name__ == "__main__":

    print(sys.argv)
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    mask_fragments = False
    mask_file = None
    mask_dataset = None
    epsilon_agglomerate = 0
    use_mahotas = False

    for key in run_config:
        globals()['%s' % key] = run_config[key]

    '''Illaria TODO
        1. Check for different thresholds
            make them daisy.Parameters
            see the plotting script for these parameters and thesholds: https://github.com/htem/segway/tree/master/synapse_evaluation
    '''

    db_client = pymongo.MongoClient(db_host)
    db = db_client[db_name]

    completion_db = db[completion_db_name]
    print("db_name: ", db_name)
    print("db_host: ", db_host)
    print("db collection names: ", db_col_name_syn, db_col_name_sf)
    
    print("super_fragments_file: ", super_fragments_file)
    print("super_fragments_dataset: ", super_fragments_dataset)
    print("syn_indicator_file: ", syn_indicator_file)
    print("syn_indicator_dataset: ", syn_indicator_dataset)
    print("syn_dir_file: ", syn_dir_file)
    print("syn_dir_dataset: ", syn_dir_dataset)
    print("score_threshold: ", score_threshold)


    parameters = detection.SynapseExtractionParameters(
        extract_type=extract_type,
        cc_threshold=cc_threshold,
        loc_type=loc_type,
        score_thr=score_threshold,
        score_type=score_type,
        nms_radius=None
    )

    print("WORKER: Running with context %s"%os.environ['DAISY_CONTEXT'])
    client_scheduler = daisy.Client()

    syn_db = SynapseDatabase(db_name, db_host, db_col_name_syn,
                 mode='r+')
    # print(syn_db.read_synapses())
    # for syn in syn_db.read_synapses():
        # print(syn)

    superfrag_db = SuperFragmentDatabase(db_name, db_host, db_col_name_sf,
                 mode='r+')
    ind_pred_ds = daisy.open_ds(syn_indicator_file, syn_indicator_dataset, 'r') 
    dir_pred_ds = daisy.open_ds(syn_dir_file, syn_dir_dataset, 'r')
    segment_ds = daisy.open_ds(super_fragments_file, super_fragments_dataset, 'r')

    while True:
        block = client_scheduler.acquire_block()
        if block is None:
            break

        logging.info("Running synapse extraction for block %s" % block)

        synapses = extract_synapses(ind_pred_ds,
                                    dir_pred_ds,
                                    segment_ds,
                                    parameters,
                                    block)

        syn_db.write_synapses(synapses)

        superfragments = extract_superfragments(synapses, block.write_roi)
        superfrag_db.write_superfragments(superfragments)

        testing = False
        if testing:
            # FOR TESTING PURPOSES, DON'T RETURN THE BLOCK
            # AND JUST QUIT

            time.sleep(1)
            sys.exit(1)

        # write block completion
        document = {
            'block_id': block.block_id,
        }
        completion_db.insert(document)

        client_scheduler.release_block(block, ret=0)

    print("NUM SYNAPSES: ", syn_db.synapses.count())
    print("NUM SUPERFRAGMENTS: ", superfrag_db.superfragments.count())
