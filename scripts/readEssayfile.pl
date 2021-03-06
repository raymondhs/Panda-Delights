#!/usr/bin/perl -w

##
# Author : Muthu Kumar C
# Script opening templates creait to Min-Yen Kan <knmnyn@gmail.com>
##

require 5.0;

use strict;
use Getopt::Long;
use FindBin;
use utf8;
use utf8::all;

my $path;	# Path to binary directory

BEGIN 
{
	if ($FindBin::Bin =~ /(.*)/) 
	{
		$path  = $1;
	}
}

use lib "$path/../lib";
use FeatureExtraction;

### USER customizable section
$0 =~ /([^\/]+)$/; my $progname = $1;
my $outputVersion = "1.0";
### END user customizable section

sub License 
{
	#print STDERR "# Copyright 2014 \251 Muthu Kumar C\n";
}

sub Help 
{
	print STDERR "Usage: $progname -h\t[invokes help]\n";
  	print STDERR "       $progname [-q quiet -force]\n";
	print STDERR "Options:\n";
	print STDERR "\t-q    \t Quiet Mode (don't echo license).\n";
	print STDERR "\t-force\t Replace all cached items with fresh results.\n";
}

my $help	= 0;
my $quite	= 0;
my $datahome = $path."/../data";
my $inputpath = $ARGV[0];
my $outputpath = $ARGV[1];

$help = 1 unless GetOptions(
				'h' 		=>	\$help,
				'q' 		=>	\$quite
			);
		
if ( $help)
{
	Help();
	exit(0);
}

if (!$quite) 
{
	License();
}

#open( CSV, "<", "../data/original/essays.csv") 
open( CSV, "<", $ARGV[0])
	or die "Cannot open file at $datahome $!";

#open( my $TITLE1, ">", $datahome."/sample/essay_title.csv") 
#	or die "Cannot open file at $datahome $!";

#open( my $TITLE2, ">", $datahome."/sample/postags/essay_title_tagged.csv") 
#	or die "Cannot open file at $datahome $!";
	
#open( my $ESSAY2, ">", "../data/essay_tagged.csv") 
open( my $ESSAY2, ">", $ARGV[1]) 
	or die "Cannot open file at $datahome $!";
	
#my $header = <CSV>;
#print $header;
while(<CSV>){
	my @fields = split(/,/,$_);
	my $project_id = $fields[0];
	#my $essay_title = $fields[2];
	
	#print $TITLE1 "$project_id".","."$essay_title\n";
	#printTags($project_id,$TITLE2,$essay_title);
	my $essay = @fields[1..$#fields];
	
	#replace first & last quote
	$essay =~ s/^\"//;
	$essay =~ s/\"$//;
	$essay =~ s/\"+/\"/;
	
	printTags($project_id,$ESSAY2,$essay);
}

sub printTags{
	my ($project_id, $fh, $text) = @_;
	my $tagged_sentences = FeatureExtraction::extractFeatures($text,1,0);
	print $fh "$project_id".",";
	
	if(keys %$tagged_sentences ==  0){
		print $fh "NULL";
	}
	
	foreach my $sid (sort {$a <=> $b} keys %$tagged_sentences){
		my $tagged = $tagged_sentences->{$sid};
        #my $tags = s/>(.*?)<\/(.*?)>/>/g;
		$tagged =~ s/(\<.*?\>)(.*?)\<\/.*?\>/$1/g;
		print $tagged;
		print " ";
		print $fh $tagged;
        print $fh " ";
	}	
	print $fh "\n";
    print "\n";
}
