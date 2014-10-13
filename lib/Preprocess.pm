package Preprocess;

# Configuration
use strict;
use warnings;

# Dependencies
use Lingua::EN::Sentence qw( get_sentences add_acronyms );
use Lingua::EN::Tokenizer::Offsets qw(token_offsets get_tokens);
use Lingua::EN::Tagger;

###
# Just a small tool to handle the cache of downloaded items
###
# Initialization
sub getTokens{
	my($sentences, $strictness) = @_;
	my $tokens;

	if( $strictness eq 1 ){
		my $splitter = new Lingua::EN::Splitter;
		my @words = undef;
		#print @$sentences;
		push (@words, $splitter->words($_));

		foreach my $word (@words){
			foreach my $w (@$word){
				push @{$tokens}, $w;
			}
		}		
	}
	else{
			push @{$tokens}, @{get_tokens($_)} ;
	}
	
	return $tokens;
}

sub getSentences{
	my ( $text ) = @_;
	my $sentences = get_sentences($text);
	return $sentences;
}

sub splitParentheses{
	my ($s) = @_;
	my @new_sentences;
	foreach (@$s){
		if( my @matches = $_ =~ /\((\s*[A-Za-z0-9].*?)\)/g ){
			$_ =~ s/\((\s*[A-Za-z0-9].*?)\)/ /g;
			foreach (@matches){
				if ( $_ !~ /(\s*[0-9].*?)/ ){
					#print "--$_\n";
					push @new_sentences,$_;
				}
			}
		}
	}
	return \@new_sentences;
}
	
sub removeOrphanCharacters{
	my($tokens) = @_;
	my @tokens_large;
	
	foreach (@$tokens){
		$_ =~ s/\s*//g;
		if(length ($_) >  1){
			push @tokens_large,$_;
		}
	}
	return \@tokens_large;
}

sub replaceURL{
	my( $sentences ) = @_;	
	$sentences =~ s/https?\:\/\/[a-zA-Z0-9][a-zA-Z\.\_\?\=\/0-9\%\-\~\&]+/<URL>/g;
	return $sentences;
}

sub getURLs{
	my( $sentences ) = @_;
	my @urls = $sentences =~ /URL/g;
	return \@urls;
}

sub removePunctuations{
	my ($sentences) = @_;
	foreach my $sentence (@$sentences){
		$sentence =~ s/[\"\,\'\?\!\_\&\=\:\\\/\<\>\(\)\[\]\{\}\%\@\#\!\*\+\-\^\.]//g;
	}
	return $sentences;
}

sub fixApostrophe{
	my ($sentences) = @_;

	foreach my $sentence (@$sentences){
		$sentence =~ s/([Nn])\'([tT])/$1$2/g;
		$sentence =~ s/([Ss])\'([sS])/$1$2/g;
		$sentence =~ s/([eE])\'([sS])/$1$2/g;
	}
	
	return $sentences;
}

sub removeMarkers{
	my ($sentences) = @_;
	foreach my $sentence (@$sentences) {
		$sentence =~ s/\<PAR\>//g;
		$sentence =~ s/\<MATH\>/ MATH /g;
		$sentence =~ s/\<MATH_FUNC\>/ MATHFUNC /g;
		$sentence =~ s/\<TIME\_REF\>/ TIMEREF /g;
		$sentence =~ s/\<URL\>/ URL /g;
	}
	return $sentences;
}

sub getQuotes{
	my( $sentences ) = @_;
	my @quotes;
	foreach my $sentence (@$sentences) {
		if ($sentence =~ /\"(.*\s?.*)?\"/){
			push @quotes,$1;
		}
	}
	return \@quotes;
}


sub replaceNumbers{
	my ($text) = @_;
	$text =~ s/[0-9]+\.\?\%/<NUMBER>/g ;
	return $text;
}

sub replaceTimeReferences{
	my ($text) = @_;
	$text =~ s/[0-9][0-9]?:[0-9][0-9]?/<TIME_REF>/g ;
	return $text;
}

sub replaceMath{
	my ($text) = @_;
	$text =~ s/\$\$.*?\$\$/ <MATH> /g ;
	$text =~ s/[a-zA-Z][a-zA-Z0-9]*\'?\(([a-zA-Z],)*[a-zA-Z]\)/<MATH_FUNC>/g;
	$text =~ s/[a-zA-Z][a-zA-Z0-9]*\'?\((<MATH_FUNC>([,=]|\s+))*<MATH_FUNC>\)/<MATH_FUNC>/g;
	$text =~ s/\(.*\(.*?\=.*?\)\)/ <MATH> /g ;
	return $text;
}

sub normalizeParaMarker{
	my ($text) = @_;
	$text =~ s/<PAR>\s*<PAR>/<PAR>/g;
	$text =~ s/<PAR>/ <PAR> /g;
	return $text;
}

sub normalizeSpace{
	my ($text) = @_;
	$text =~ s/\s+/ /g;
	return $text;
}

sub normalizePeriods{
	my ($text) = @_;
	$text =~ s/\.+/\./g;
	$text =~ s/[\?\!]\.+/\./g;
	return $text;
}
1;
