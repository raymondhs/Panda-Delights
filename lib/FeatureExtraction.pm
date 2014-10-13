package FeatureExtraction;

# Configuration
use strict;
use warnings;

##
#
# Author : Muthu Kumar C
# Recreate thread from database
# Created in Mar, 2014
#
##

# External libraries
# Dependencies
use FindBin;
use Getopt::Long;
use Lingua::EN::Tokenizer::Offsets qw( token_offsets tokenize get_tokens );

my $path;	# Path to binary directory

BEGIN 
{
	if ($FindBin::Bin =~ /(.*)/) 
	{
		$path  = $1;
	}
}

use lib "$path/../lib";
#local libraries
use Preprocess;

sub merge_hashes{
	my ($tohash, $fromhash) =@_;
	
	foreach my $fromkey ( keys %{$fromhash} ){
		if( !exists $tohash->{$fromkey} ){
			$tohash->{$fromkey} = $fromhash->{$fromkey};
		}
	}
	return $tohash;
}

sub extractFeatures{
	my($text, $pos, $debug) = @_;
	
	if( !defined $text || $text eq '' ) { return; }
	
	# features to extract
	my $hascite = 0;
	my $hasurl = 0;
	my $textlength = 0;
	my $hastimeref = 0;
	my $posbigrams;
	
	#length of post in words excluding punctuations

	$text = Preprocess::replaceURL($text);
	$text = Preprocess::replaceNumbers($text);
	$text = Preprocess::normalizePeriods($text);
	$text = Preprocess::normalizeSpace($text);
	
	my $sentences = Preprocess::getSentences($text);
	# TODO find if sentence is question. 
	# this is an interesting feature
	
	# quotation/citation features
	my $quotes = Preprocess::getQuotes($sentences);
	
	my $extracted_sentences = Preprocess::splitParentheses($sentences);
	foreach (@$extracted_sentences){
		push \@$sentences, $_;
	}
		
	#$sentences = Preprocess::fixApostrophe($sentences);
	#$sentences = Preprocess::removePunctuations( $sentences);

	#POS tagging
	my $tagged_sentences = undef;
	
	if ($pos){
		$tagged_sentences = getPOStagged($sentences,$debug);
		#Create POS bigrams
		#$posbigrams = createPOSBigrams($tagged_sentences);
	}
	
	#Extracting ngram features respecting sentence boundaries
	#$sentences = Preprocess::fixApostrophe($sentences);
	#$sentences = Preprocess::removePunctuations($sentences);
	
	# Lowercase and tokenize to words
	
	return $tagged_sentences;
}

sub createPOSBigrams{
	my ($tagged_sentences) = @_;
	my %posbigrams = ();
	foreach my $sid ( sort{$a <=> $b} (keys %{$tagged_sentences}) ){
		my $sentence = $tagged_sentences->{$sid};
		$sentence =~ s/>(.*?)</ /g;
		$sentence =~ s/[\"\,\'\?\!\_\&\=\:\\\/\<\>\(\)\[\]\{\}\%\@\#\!\*\+\-\^\.]/ /g;
		$sentence =~ s/\s+((pp)|(ppc)|(ppd)|(ppl)|(ppr)|(pps)|(lrb)|(rrb)|(sym))\s+/ /g;
		$sentence = Preprocess::normalizeSpace($sentence);
		
		my $bigraminputtokens = get_tokens($sentence);
		
		my $tokenid = 0;
		foreach my $t (@$bigraminputtokens){
			$t = $t.$tokenid;
			$tokenid++;			
		}
		
		my $temp = getBigramsPos($bigraminputtokens);
		
		my %temp2 = ();
		foreach my $bi ( sort{$temp->{$a} <=> $temp->{$b}} (keys %{$temp}) ){
			my $value = $temp->{$bi};
			$bi =~ s/[0-9]//g;
			$temp2{$bi} = $value;
		}
		
		$posbigrams{$sid} = \%temp2;
	}
	return \%posbigrams;
}

sub getBigramsPos{
	my ($gram_input) = @_;
	my %bigram_positions = ();
	
	my $pos = 0;
	foreach my $gram (@$gram_input){
		# Failsafe for end of array
		if ($pos+1 >= scalar (@$gram_input)){ last; }
		
		my $bigram = $gram . " " . (@$gram_input)[$pos+1];
		$bigram_positions{$bigram} = $pos;
		$pos++;
	}
	
	return \%bigram_positions;
}

sub getPOStagged{
	my($sentences,$debug) = @_;
	my %tagged_sentences = ();
	my $sid = 1;
	foreach (@$sentences){
			# Create a parser object
			my $posTagger = new Lingua::EN::Tagger;

			# Add part of speech tags to a text
			my $tagged_text = $posTagger->add_tags($_);
			
			if ($debug){		print $tagged_text."\n";	}
			
			$tagged_sentences{$sid} = $tagged_text;
			$sid ++;
	}
	return \%tagged_sentences;
}

1;