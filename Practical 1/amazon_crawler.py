# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 19:08:25 2014

@author: vincentli2010
"""

import amazonproduct
config = {
    'access_key': 'AKIAIQDB4PGWWRQZ5CNA',
    'secret_key': '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',
    'associate_tag': 'lycugus-20',
    'locale': 'us'
}
api = amazonproduct.API(cfg=config)
result = api.item_lookup(
    SearchIndex='Books', IdType='ISBN', ItemId='0002005018',
    ResponseGroup='Reviews', TruncateReviewsAt=256, IncludeReviewsSummary=False)

result = api.item_lookup(
    SearchIndex='Books', IdType='ISBN', ItemId='0002005018',
    ResponseGroup='SalesRank')
sales_rank = result.Items.Item.SalesRank    

"""
for root in api.item_search('Books', Publisher='Apress',
                            ResponseGroup='Large'):

    # extract paging information
    total_results = root.Items.TotalResults.pyval
    total_pages = root.Items.TotalPages.pyval
    try:
        current_page = root.Items.Request.ItemSearchRequest.ItemPage.pyval
    except AttributeError:
        current_page = 1

    print 'page %d of %d' % (current_page, total_pages)

    #~ from lxml import etree
    #~ print etree.tostring(root, pretty_print=True)

    nspace = root.nsmap.get(None, '')
    books = root.xpath('//aws:Items/aws:Item', 
                         namespaces={'aws' : nspace})

    for book in books:
        print book.ASIN,
        if hasattr(book.ItemAttributes, 'Author'): 
            print unicode(book.ItemAttributes.Author), ':', 
        print unicode(book.ItemAttributes.Title),
        if hasattr(book.ItemAttributes, 'ListPrice'): 
            print unicode(book.ItemAttributes.ListPrice.FormattedPrice)
        elif hasattr(book.OfferSummary, 'LowestUsedPrice'):
            print u'(used from %s)' % book.OfferSummary.LowestUsedPrice.FormattedPrice
"""